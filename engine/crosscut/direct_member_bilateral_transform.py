"""Two source-bound camera proposals for an authenticated member hard cut.

This adds no identity evidence and grants no rendering authority. Both endpoint
cameras must subsequently pass the dense planner's complete visible windows.
The single-sided resolver remains the first choice and is not changed here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Sequence

from .direct_member_transform import (
    DirectMemberSourceFrameGeometry, NormalizedBox, _validated_box,
    attention_mismatch, box_center, box_visible_fraction,
    face_geometry_mismatch, project_box,
)

_AUTHENTICATED_BASES = frozenset({
    "verified_official_fancam_focal_face_correspondence",
    "frozen_source_focal_body_track_geometry",
})


@dataclass(frozen=True, slots=True)
class DirectMemberBoundaryCamera:
    source_index: int
    zoom: float
    center_x: float
    center_y: float
    source_geometry: DirectMemberSourceFrameGeometry
    face_visible_fraction: float
    subject_geometry_basis: str | None
    subject_visible_fraction: float | None
    maximum_upscale_ratio: float

    def __post_init__(self) -> None:
        if (isinstance(self.source_index, bool) or not isinstance(self.source_index, int)
                or self.source_index < 0):
            raise ValueError("paired member camera source index is invalid")
        values = (self.zoom, self.center_x, self.center_y, self.face_visible_fraction,
                  self.maximum_upscale_ratio)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError("paired member camera scalars must be finite")
        if not isinstance(self.source_geometry, DirectMemberSourceFrameGeometry):
            raise ValueError("paired member camera native geometry is invalid")
        if not 1.0 <= self.maximum_upscale_ratio <= 1.15:
            raise ValueError("paired member camera upscale allowance is invalid")
        if not 1.0 <= self.zoom <= self.source_geometry.lossless_zoom_capacity * self.maximum_upscale_ratio + 1e-9:
            raise ValueError("paired member camera exceeds native resolution allowance")
        half = .5 / self.zoom
        if any(not half - 1e-9 <= v <= 1.0 - half + 1e-9 for v in (self.center_x, self.center_y)):
            raise ValueError("paired member camera exposes pixels outside the source")
        if not 0.0 <= self.face_visible_fraction <= 1.0 + 1e-9:
            raise ValueError("paired member camera visibility is invalid")
        if self.subject_visible_fraction is not None and (
                not self.subject_geometry_basis or not math.isfinite(self.subject_visible_fraction)
                or not 0.0 <= self.subject_visible_fraction <= 1.0 + 1e-9):
            raise ValueError("paired member subject visibility is invalid")

    @property
    def center_shift(self) -> float:
        return math.hypot(self.center_x - .5, self.center_y - .5)

    @property
    def upscale_ratio(self) -> float:
        return max(1.0, self.zoom / self.source_geometry.lossless_zoom_capacity)

    def project(self, box: NormalizedBox) -> NormalizedBox:
        return project_box(box, zoom=self.zoom, center_x=self.center_x, center_y=self.center_y)

    def to_dict(self) -> dict[str, Any]:
        return dict(source_index=self.source_index, zoom=self.zoom,
                    center_xy=[self.center_x, self.center_y],
                    source_geometry=self.source_geometry.to_dict(),
                    face_visible_fraction=self.face_visible_fraction,
                    subject_geometry_basis=self.subject_geometry_basis,
                    subject_visible_fraction=self.subject_visible_fraction,
                    upscale_ratio=self.upscale_ratio,
                    maximum_upscale_ratio=self.maximum_upscale_ratio)


@dataclass(frozen=True, slots=True)
class DirectMemberBilateralTransformRepair:
    outgoing: DirectMemberBoundaryCamera
    incoming: DirectMemberBoundaryCamera
    outgoing_box: NormalizedBox
    incoming_box: NormalizedBox
    alignment_geometry_basis: str
    repaired_severe_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all(isinstance(camera, DirectMemberBoundaryCamera) for camera in self.cameras):
            raise ValueError("paired member repair needs both typed source cameras")
        object.__setattr__(self, 'outgoing_box', _validated_box(self.outgoing_box))
        object.__setattr__(self, 'incoming_box', _validated_box(self.incoming_box))
        object.__setattr__(self, 'repaired_severe_flags', tuple(self.repaired_severe_flags))
        if self.alignment_geometry_basis not in _AUTHENTICATED_BASES:
            raise ValueError("paired member repair needs authenticated focal geometry")
        if self.outgoing.source_index == self.incoming.source_index:
            raise ValueError("paired member repair needs two distinct sources")
        if (self.outgoing.source_geometry.output_width, self.outgoing.source_geometry.output_height) != (
                self.incoming.source_geometry.output_width, self.incoming.source_geometry.output_height):
            raise ValueError("paired member cameras disagree on output dimensions")
        if any(flag not in {"attention_position", "attention_scale"} for flag in self.repaired_severe_flags):
            raise ValueError("paired camera cannot repair pose or motion evidence")

    @property
    def cameras(self) -> tuple[DirectMemberBoundaryCamera, DirectMemberBoundaryCamera]:
        return self.outgoing, self.incoming

    @property
    def transformed_side(self) -> str:
        return "both"

    @property
    def residual_attention_position_mismatch(self) -> float:
        return attention_mismatch(self.outgoing.project(self.outgoing_box),
                                  self.incoming.project(self.incoming_box))[0]

    @property
    def residual_attention_scale_mismatch(self) -> float:
        return attention_mismatch(self.outgoing.project(self.outgoing_box),
                                  self.incoming.project(self.incoming_box))[1]

    @property
    def transformed_face_geometry_mismatch(self) -> float:
        return face_geometry_mismatch(self.outgoing.project(self.outgoing_box),
                                      self.incoming.project(self.incoming_box))

    @property
    def improvement_fraction(self) -> float:
        baseline = face_geometry_mismatch(self.outgoing_box, self.incoming_box)
        return (baseline - self.transformed_face_geometry_mismatch) / baseline if baseline > 1e-9 else 0.0

    @property
    def upscale_ratio(self) -> float:
        return max(camera.upscale_ratio for camera in self.cameras)

    def with_repaired_severe_flags(self, flags: Sequence[str]) -> DirectMemberBilateralTransformRepair:
        return replace(self, repaired_severe_flags=tuple(flags))

    def to_dict(self) -> dict[str, Any]:
        raw = attention_mismatch(self.outgoing_box, self.incoming_box)
        return {
            "contract": "official_member_paired_resolution_bound_subject_transform_v1",
            "required": True, "transformed_side": "both",
            "cameras": {"outgoing": self.outgoing.to_dict(), "incoming": self.incoming.to_dict()},
            "endpoint_boxes_xywh": [list(self.outgoing_box), list(self.incoming_box)],
            "raw_attention_mismatch": dict(position=raw[0], scale=raw[1]),
            "residual_attention_mismatch": dict(position=self.residual_attention_position_mismatch,
                                                scale=self.residual_attention_scale_mismatch),
            "face_geometry_mismatch": dict(before=face_geometry_mismatch(self.outgoing_box, self.incoming_box),
                                           after=self.transformed_face_geometry_mismatch,
                                           improvement_fraction=self.improvement_fraction),
            "alignment_geometry_basis": self.alignment_geometry_basis,
            "repaired_severe_flags": list(self.repaired_severe_flags),
            "upscale_ratio": self.upscale_ratio, "lossless": self.upscale_ratio <= 1.0 + 1e-9,
            "dense_every_output_frame_revalidation_required": True,
            "paired_atomic_publication_required": True,
            "rendered_quality_approved": False, "identity_inference_used": False,
        }


def _paired_camera_centers(box, target_box, zoom, maximum_center_shift):
    # Same rectangular viewport and radial pan bound as the dense camera.
    half = .5 / zoom
    desired = tuple(a - (b - .5) / zoom for a, b in zip(box_center(box), box_center(target_box)))
    bounded = tuple(min(1.0 - half, max(half, v)) for v in desired)
    distance = math.hypot(bounded[0] - .5, bounded[1] - .5)
    if distance > maximum_center_shift + 1e-12:
        bounded = tuple(.5 + (v - .5) * maximum_center_shift / distance for v in bounded)
    return (bounded, (.5, .5)) if bounded != (.5, .5) else (bounded,)


def find_bilateral_boundary_transform_repair(
    *, outgoing_face_box, incoming_face_box, outgoing_source_index,
    incoming_source_index, outgoing_source_geometry, incoming_source_geometry,
    outgoing_subject_box, incoming_subject_box, outgoing_subject_basis,
    incoming_subject_basis, maximum_zoom, maximum_center_shift,
    maximum_boundary_center_distance, maximum_residual_center_distance,
    maximum_residual_attention_position_mismatch,
    maximum_residual_attention_scale_mismatch, minimum_improvement_fraction,
    minimum_face_visible_fraction, legacy_face_only_maximum_zoom,
    legacy_face_only_maximum_center_shift, minimum_subject_visible_fraction,
    maximum_upscale_ratio, alignment_geometry_basis,
) -> DirectMemberBilateralTransformRepair | None:
    """Bounded endpoint feasibility, using the dense 33-height/9x9 family.

    Only previously unresolved authenticated focal pairs reach this function.
    It does not read media, relax matching limits or remove motion/pose checks.
    """
    if (isinstance(maximum_upscale_ratio, bool) or not math.isfinite(maximum_upscale_ratio)
            or not 1.0 <= maximum_upscale_ratio <= 1.15):
        raise ValueError("paired member camera upscale allowance is invalid")
    if alignment_geometry_basis not in _AUTHENTICATED_BASES or outgoing_source_index == incoming_source_index:
        return None
    left, right = _validated_box(outgoing_face_box), _validated_box(incoming_face_box)
    if math.dist(box_center(left), box_center(right)) > maximum_boundary_center_distance + 1e-9:
        return None
    if min(box_visible_fraction(box, zoom=1.0, center_x=.5, center_y=.5)
           for box in (left, right)) + 1e-9 < minimum_face_visible_fraction:
        return None
    geometries = (outgoing_source_geometry, incoming_source_geometry)
    baseline = face_geometry_mismatch(left, right)
    if baseline <= 1e-9:
        return None
    # Equal rendered heights preserve the two boxes' aspect-ratio difference.
    # This necessary condition avoids a grid search that cannot fix area scale.
    fixed_scale = min(1.0, abs(math.log((left[2] / left[3]) / (right[2] / right[3]))) / math.log(3.2))
    if fixed_scale > maximum_residual_attention_scale_mismatch + 1e-9:
        return None
    for allowance in dict.fromkeys((1.0, maximum_upscale_ratio)):
        limits = tuple(min(maximum_zoom, geometry.lossless_zoom_capacity * allowance) for geometry in geometries)
        low, high = max(left[3], right[3]), min(box[3] * limit for box, limit in zip((left, right), limits))
        if high + 1e-9 < low:
            continue
        best = None
        seen = set()
        for step in range(33):
            height = low + max(0.0, high - low) * step / 32.0
            for xi in range(-4, 5):
                for yi in range(-4, 5):
                    target = (.50 + .04 * xi - .05, .50 + .04 * yi - height / 2, .10, height)
                    sides = []
                    for box, index, geometry, subject, basis in zip(
                        (left, right), (outgoing_source_index, incoming_source_index), geometries,
                        (outgoing_subject_box, incoming_subject_box), (outgoing_subject_basis, incoming_subject_basis),
                    ):
                        zoom = max(1.0, height / box[3])
                        states = []
                        for cx, cy in _paired_camera_centers(box, target, zoom, maximum_center_shift):
                            visible = box_visible_fraction(box, zoom=zoom, center_x=cx, center_y=cy)
                            if visible + 1e-9 < minimum_face_visible_fraction:
                                continue
                            widened = zoom > legacy_face_only_maximum_zoom + 1e-9 or math.hypot(cx - .5, cy - .5) > legacy_face_only_maximum_center_shift + 1e-9
                            subject_visible = None
                            if widened:
                                if subject is None or not basis:
                                    continue
                                subject_visible = box_visible_fraction(subject, zoom=zoom, center_x=cx, center_y=cy)
                                if subject_visible + 1e-9 < minimum_subject_visible_fraction:
                                    continue
                            states.append(DirectMemberBoundaryCamera(index, zoom, cx, cy, geometry,
                                          visible, basis if widened else None, subject_visible, allowance))
                        sides.append(states)
                    for a in sides[0]:
                        for b in sides[1]:
                            key = (a.zoom, a.center_x, a.center_y, b.zoom, b.center_x, b.center_y)
                            if key in seen:
                                continue
                            seen.add(key)
                            l, r = a.project(left), b.project(right)
                            position, scale = attention_mismatch(l, r)
                            score = face_geometry_mismatch(l, r)
                            if (position > maximum_residual_attention_position_mismatch + 1e-9
                                    or position * .42 > maximum_residual_center_distance + 1e-9
                                    or scale > maximum_residual_attention_scale_mismatch + 1e-9
                                    or score > baseline * (1.0 - minimum_improvement_fraction) + 1e-9):
                                continue
                            effort = a.zoom + b.zoom - 2.0 + a.center_shift + b.center_shift
                            ranking = (score, effort, max(a.upscale_ratio, b.upscale_ratio))
                            if best is None or ranking < best[0]:
                                best = (ranking, a, b)
        if best is not None:
            return DirectMemberBilateralTransformRepair(best[1], best[2], left, right, alignment_geometry_basis)
    return None
