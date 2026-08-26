"""Resolution-bound geometry for direct-member cut repair.

Official target-member fancams may use a short crop/zoom/pan ramp to match a
face across a hard cut. A repair is admissible only when the source retains
enough decoded pixels for the enlarged working frame and the transformed face
and, for widened moves, the complete subject remain inside the camera window.

All inputs and persisted outputs are normalized geometry and scalar capacity;
this module never decodes pixels and never performs identity inference.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Sequence


NormalizedBox = tuple[float, float, float, float]


def _validated_box(box: Sequence[float]) -> NormalizedBox:
    if len(box) != 4:
        raise ValueError("direct member transform box needs four values")
    x, y, width, height = tuple(float(value) for value in box)
    if any(not math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError("direct member transform box must be finite")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("direct member transform box must have positive area")
    return (x, y, width, height)


def _even_floor(value: float) -> int:
    return max(0, int(math.floor(value)) // 2 * 2)


@dataclass(frozen=True, slots=True)
class DirectMemberSourceFrameGeometry:
    """One source after its centered aspect-fill crop, before final scaling."""

    source_width: int
    source_height: int
    output_width: int
    output_height: int
    fill_crop_xywh: NormalizedBox
    effective_width_pixels: int
    effective_height_pixels: int
    lossless_zoom_capacity: float

    def __post_init__(self) -> None:
        integers = (
            self.source_width,
            self.source_height,
            self.output_width,
            self.output_height,
            self.effective_width_pixels,
            self.effective_height_pixels,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in integers
        ):
            raise ValueError("direct member source frame dimensions are invalid")
        _validated_box(self.fill_crop_xywh)
        if (
            not math.isfinite(self.lossless_zoom_capacity)
            or self.lossless_zoom_capacity <= 0.0
        ):
            raise ValueError("direct member lossless zoom capacity is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_resolution": [self.source_width, self.source_height],
            "output_resolution": [self.output_width, self.output_height],
            "portrait_fill_crop_xywh": list(self.fill_crop_xywh),
            "effective_even_resolution": [
                self.effective_width_pixels,
                self.effective_height_pixels,
            ],
            "lossless_zoom_capacity": self.lossless_zoom_capacity,
            "capacity_contract": (
                "even_working_frame_not_larger_than_post_fill_source_v1"
            ),
        }


def direct_member_source_frame_geometry(
    source_size: Sequence[int],
    *,
    output_size: tuple[int, int] = (1080, 1920),
) -> DirectMemberSourceFrameGeometry:
    """Resolve aspect-fill pixels available to a 9:16 editorial camera."""

    if len(source_size) != 2:
        raise ValueError("direct member source size needs width and height")
    source_width, source_height = source_size
    output_width, output_height = output_size
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        for value in (
            source_width,
            source_height,
            output_width,
            output_height,
        )
    ):
        raise ValueError("direct member source/output size must be positive integers")
    source_ratio = source_width / source_height
    output_ratio = output_width / output_height
    if source_ratio > output_ratio:
        crop_width = output_ratio / source_ratio
        crop = ((1.0 - crop_width) / 2.0, 0.0, crop_width, 1.0)
    elif source_ratio < output_ratio:
        crop_height = source_ratio / output_ratio
        crop = (0.0, (1.0 - crop_height) / 2.0, 1.0, crop_height)
    else:
        crop = (0.0, 0.0, 1.0, 1.0)
    effective_width = _even_floor(source_width * crop[2])
    effective_height = _even_floor(source_height * crop[3])
    if effective_width <= 0 or effective_height <= 0:
        raise ValueError("direct member portrait-fill crop has no pixels")
    capacity = min(
        effective_width / output_width,
        effective_height / output_height,
    )
    return DirectMemberSourceFrameGeometry(
        source_width=source_width,
        source_height=source_height,
        output_width=output_width,
        output_height=output_height,
        fill_crop_xywh=crop,
        effective_width_pixels=effective_width,
        effective_height_pixels=effective_height,
        lossless_zoom_capacity=capacity,
    )


def map_box_to_portrait_fill(
    box: Sequence[float],
    geometry: DirectMemberSourceFrameGeometry,
) -> NormalizedBox:
    """Map decoded-source coordinates into the centered aspect-fill domain."""

    x, y, width, height = _validated_box(box)
    crop_x, crop_y, crop_width, crop_height = geometry.fill_crop_xywh
    return (
        (x - crop_x) / crop_width,
        (y - crop_y) / crop_height,
        width / crop_width,
        height / crop_height,
    )


def box_center(box: Sequence[float]) -> tuple[float, float]:
    x, y, width, height = _validated_box(box)
    return (x + width / 2.0, y + height / 2.0)


def project_box(
    box: Sequence[float],
    *,
    zoom: float,
    center_x: float,
    center_y: float,
) -> NormalizedBox:
    x, y, width, height = _validated_box(box)
    return (
        (x - center_x) * zoom + 0.5,
        (y - center_y) * zoom + 0.5,
        width * zoom,
        height * zoom,
    )


def box_visible_fraction(
    box: Sequence[float],
    *,
    zoom: float,
    center_x: float,
    center_y: float,
) -> float:
    x, y, width, height = _validated_box(box)
    half = 0.5 / zoom
    camera_left = center_x - half
    camera_top = center_y - half
    camera_right = center_x + half
    camera_bottom = center_y + half
    intersection_width = max(
        0.0,
        min(camera_right, x + width) - max(camera_left, x),
    )
    intersection_height = max(
        0.0,
        min(camera_bottom, y + height) - max(camera_top, y),
    )
    return intersection_width * intersection_height / (width * height)


def attention_mismatch(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float]:
    left_box = _validated_box(left)
    right_box = _validated_box(right)
    left_center = box_center(left_box)
    right_center = box_center(right_box)
    position = min(
        1.0,
        math.hypot(
            left_center[0] - right_center[0],
            left_center[1] - right_center[1],
        )
        / 0.42,
    )
    left_area = max(1e-9, left_box[2] * left_box[3])
    right_area = max(1e-9, right_box[2] * right_box[3])
    scale = min(1.0, abs(math.log(left_area / right_area)) / math.log(3.2))
    return (position, scale)


def face_geometry_mismatch(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    left_box = _validated_box(left)
    right_box = _validated_box(right)
    return abs(math.log(left_box[3] / right_box[3])) + math.hypot(
        box_center(left_box)[0] - box_center(right_box)[0],
        box_center(left_box)[1] - box_center(right_box)[1],
    )


@dataclass(frozen=True, slots=True)
class DirectMemberBoundaryTransformRepair:
    """A selector-authorized repair that the dense planner must reproduce."""

    transformed_side: str
    transformed_source_index: int
    zoom: float
    center_x: float
    center_y: float
    source_geometry: DirectMemberSourceFrameGeometry
    raw_attention_position_mismatch: float
    raw_attention_scale_mismatch: float
    residual_attention_position_mismatch: float
    residual_attention_scale_mismatch: float
    baseline_face_geometry_mismatch: float
    transformed_face_geometry_mismatch: float
    improvement_fraction: float
    face_visible_fraction: float
    subject_geometry_basis: str | None
    subject_visible_fraction: float | None
    alignment_geometry_basis: str = (
        "unambiguous_dominant_face_bbox_geometry_only"
    )
    repaired_severe_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.transformed_side not in {"outgoing", "incoming"}:
            raise ValueError("direct member repair side is invalid")
        if (
            isinstance(self.transformed_source_index, bool)
            or not isinstance(self.transformed_source_index, int)
            or self.transformed_source_index < 0
        ):
            raise ValueError("direct member repair source index is invalid")
        numeric = (
            self.zoom,
            self.center_x,
            self.center_y,
            self.raw_attention_position_mismatch,
            self.raw_attention_scale_mismatch,
            self.residual_attention_position_mismatch,
            self.residual_attention_scale_mismatch,
            self.baseline_face_geometry_mismatch,
            self.transformed_face_geometry_mismatch,
            self.improvement_fraction,
            self.face_visible_fraction,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("direct member repair scalars must be finite")
        if not 1.0 < self.zoom <= self.source_geometry.lossless_zoom_capacity + 1e-9:
            raise ValueError("direct member repair exceeds lossless zoom capacity")
        if self.alignment_geometry_basis not in {
            "unambiguous_dominant_face_bbox_geometry_only",
            "aggregate_human_silhouette",
            "unambiguous_dominant_person_box_geometry_only",
        }:
            raise ValueError("direct member repair alignment basis is invalid")

    @property
    def center_shift(self) -> float:
        return math.hypot(self.center_x - 0.5, self.center_y - 0.5)

    def with_repaired_severe_flags(
        self,
        flags: Sequence[str],
    ) -> DirectMemberBoundaryTransformRepair:
        return replace(self, repaired_severe_flags=tuple(str(flag) for flag in flags))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": (
                "official_member_lossless_resolution_bound_subject_transform_v2"
            ),
            "required": True,
            "transformed_side": self.transformed_side,
            "transformed_source_index": self.transformed_source_index,
            "zoom": self.zoom,
            "center_xy": [self.center_x, self.center_y],
            "center_shift": self.center_shift,
            "source_geometry": self.source_geometry.to_dict(),
            "raw_attention_mismatch": {
                "position": self.raw_attention_position_mismatch,
                "scale": self.raw_attention_scale_mismatch,
            },
            "residual_attention_mismatch": {
                "position": self.residual_attention_position_mismatch,
                "scale": self.residual_attention_scale_mismatch,
            },
            "face_geometry_mismatch": {
                "before": self.baseline_face_geometry_mismatch,
                "after": self.transformed_face_geometry_mismatch,
                "improvement_fraction": self.improvement_fraction,
            },
            "face_visible_fraction": self.face_visible_fraction,
            "alignment_geometry_basis": self.alignment_geometry_basis,
            "alignment_geometry_visible_fraction": self.face_visible_fraction,
            "subject_geometry_basis": self.subject_geometry_basis,
            "subject_visible_fraction": self.subject_visible_fraction,
            "repaired_severe_flags": list(self.repaired_severe_flags),
            "dense_every_output_frame_revalidation_required": True,
            "identity_inference_used": False,
        }


def _camera_states(
    *,
    box: NormalizedBox,
    target_box: NormalizedBox,
    zoom: float,
    maximum_center_shift: float,
) -> tuple[tuple[float, float], ...]:
    subject_x, subject_y = box_center(box)
    target_x, target_y = box_center(target_box)
    half = 0.5 / zoom
    low = half
    high = 1.0 - half
    desired_x = subject_x - (target_x - 0.5) / zoom
    desired_y = subject_y - (target_y - 0.5) / zoom
    bounded_x = min(high, max(low, desired_x))
    bounded_y = min(high, max(low, desired_y))
    delta_x = bounded_x - 0.5
    delta_y = bounded_y - 0.5
    distance = math.hypot(delta_x, delta_y)
    allowed_shift = min(maximum_center_shift, max(0.0, 0.5 - half))
    if distance > allowed_shift + 1e-12:
        scale = allowed_shift / distance if distance > 1e-12 else 0.0
        bounded_x = 0.5 + delta_x * scale
        bounded_y = 0.5 + delta_y * scale
    panned = (bounded_x, bounded_y)
    return (panned, (0.5, 0.5)) if panned != (0.5, 0.5) else (panned,)


def find_lossless_boundary_transform_repair(
    *,
    outgoing_face_box: NormalizedBox,
    incoming_face_box: NormalizedBox,
    outgoing_source_index: int,
    incoming_source_index: int,
    outgoing_source_geometry: DirectMemberSourceFrameGeometry,
    incoming_source_geometry: DirectMemberSourceFrameGeometry,
    outgoing_subject_box: NormalizedBox | None,
    incoming_subject_box: NormalizedBox | None,
    outgoing_subject_basis: str | None,
    incoming_subject_basis: str | None,
    maximum_zoom: float,
    maximum_center_shift: float,
    maximum_boundary_center_distance: float,
    maximum_residual_center_distance: float,
    maximum_residual_attention_scale_mismatch: float,
    minimum_improvement_fraction: float,
    minimum_face_visible_fraction: float,
    legacy_face_only_maximum_zoom: float,
    legacy_face_only_maximum_center_shift: float,
    minimum_subject_visible_fraction: float,
    alignment_geometry_basis: str = (
        "unambiguous_dominant_face_bbox_geometry_only"
    ),
) -> DirectMemberBoundaryTransformRepair | None:
    """Find one source-side transform that aligns authenticated subject geometry."""

    left = _validated_box(outgoing_face_box)
    right = _validated_box(incoming_face_box)
    if min(
        box_visible_fraction(left, zoom=1.0, center_x=0.5, center_y=0.5),
        box_visible_fraction(right, zoom=1.0, center_x=0.5, center_y=0.5),
    ) + 1e-9 < minimum_face_visible_fraction:
        return None
    center_distance = math.hypot(
        box_center(left)[0] - box_center(right)[0],
        box_center(left)[1] - box_center(right)[1],
    )
    if center_distance > maximum_boundary_center_distance + 1e-9:
        return None
    raw_position, raw_scale = attention_mismatch(left, right)
    baseline = face_geometry_mismatch(left, right)
    candidates: list[DirectMemberBoundaryTransformRepair] = []
    sides = (
        (
            "outgoing",
            outgoing_source_index,
            left,
            right,
            outgoing_source_geometry,
            outgoing_subject_box,
            outgoing_subject_basis,
        ),
        (
            "incoming",
            incoming_source_index,
            right,
            left,
            incoming_source_geometry,
            incoming_subject_box,
            incoming_subject_basis,
        ),
    )
    for (
        side,
        source_index,
        box,
        target_box,
        source_geometry,
        subject_box,
        subject_basis,
    ) in sides:
        lossless_limit = min(maximum_zoom, source_geometry.lossless_zoom_capacity)
        if lossless_limit <= 1.0 + 1e-9:
            continue
        ideal_zoom = target_box[3] / box[3]
        requested_zoom = max(1.02, ideal_zoom)
        zoom = min(lossless_limit, requested_zoom)
        if zoom <= 1.0 + 1e-9:
            continue
        for center_x, center_y in _camera_states(
            box=box,
            target_box=target_box,
            zoom=zoom,
            maximum_center_shift=maximum_center_shift,
        ):
            face_visible = box_visible_fraction(
                box,
                zoom=zoom,
                center_x=center_x,
                center_y=center_y,
            )
            if face_visible + 1e-9 < minimum_face_visible_fraction:
                continue
            shift = math.hypot(center_x - 0.5, center_y - 0.5)
            widened = (
                zoom > legacy_face_only_maximum_zoom + 1e-9
                or shift > legacy_face_only_maximum_center_shift + 1e-9
            )
            mapped_subject = (
                _validated_box(subject_box) if subject_box is not None else None
            )
            subject_visible: float | None = None
            if widened:
                if mapped_subject is None or not subject_basis:
                    continue
                subject_visible = box_visible_fraction(
                    mapped_subject,
                    zoom=zoom,
                    center_x=center_x,
                    center_y=center_y,
                )
                if subject_visible + 1e-9 < minimum_subject_visible_fraction:
                    continue
            transformed_box = project_box(
                box,
                zoom=zoom,
                center_x=center_x,
                center_y=center_y,
            )
            if side == "outgoing":
                transformed_left, transformed_right = transformed_box, right
            else:
                transformed_left, transformed_right = left, transformed_box
            residual_position, residual_scale = attention_mismatch(
                transformed_left,
                transformed_right,
            )
            residual_center = math.hypot(
                box_center(transformed_left)[0] - box_center(transformed_right)[0],
                box_center(transformed_left)[1] - box_center(transformed_right)[1],
            )
            if residual_center > maximum_residual_center_distance + 1e-9:
                continue
            if residual_scale > maximum_residual_attention_scale_mismatch + 1e-9:
                continue
            transformed_mismatch = face_geometry_mismatch(
                transformed_left,
                transformed_right,
            )
            improvement = (
                (baseline - transformed_mismatch) / baseline
                if baseline > 1e-9
                else 0.0
            )
            if improvement + 1e-9 < minimum_improvement_fraction:
                continue
            candidates.append(
                DirectMemberBoundaryTransformRepair(
                    transformed_side=side,
                    transformed_source_index=source_index,
                    zoom=zoom,
                    center_x=center_x,
                    center_y=center_y,
                    source_geometry=source_geometry,
                    raw_attention_position_mismatch=raw_position,
                    raw_attention_scale_mismatch=raw_scale,
                    residual_attention_position_mismatch=residual_position,
                    residual_attention_scale_mismatch=residual_scale,
                    baseline_face_geometry_mismatch=baseline,
                    transformed_face_geometry_mismatch=transformed_mismatch,
                    improvement_fraction=improvement,
                    face_visible_fraction=face_visible,
                    subject_geometry_basis=(subject_basis if widened else None),
                    subject_visible_fraction=(subject_visible if widened else None),
                    alignment_geometry_basis=alignment_geometry_basis,
                )
            )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item.transformed_face_geometry_mismatch,
            item.zoom - 1.0,
            item.center_shift,
            item.transformed_source_index,
        ),
    )


__all__ = [
    "DirectMemberBoundaryTransformRepair",
    "DirectMemberSourceFrameGeometry",
    "attention_mismatch",
    "box_center",
    "box_visible_fraction",
    "direct_member_source_frame_geometry",
    "face_geometry_mismatch",
    "find_lossless_boundary_transform_repair",
    "map_box_to_portrait_fill",
    "project_box",
]
