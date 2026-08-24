"""FFmpeg render-plan construction for synchronized cross-edits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable

from .ive import AspectRatio
from .timewarp import TimeWarpMap


RENDER_PLAN_SOURCE_INDEX_DOMAIN = "render_plan_sources_array_index"
MAXIMUM_RENDER_TIMEWARP_RATE_DELTA = 0.02
# Joining an identity same-source run may replace several locally linear
# time-warps with one linear clock.  Keep every serialized internal endpoint
# within 1/500th of one output frame so the optimization cannot hide a real
# source-clock edit or meaningfully move a broadcaster-native camera cut.
MAXIMUM_COALESCED_INTERNAL_CLOCK_ERROR_FRAMES = 0.002
MAXIMUM_COALESCED_SOURCE_GAP_SECONDS = 1e-6


@dataclass(frozen=True, slots=True)
class NormalizedCrop:
    """A normalized crop rectangle in source coordinates."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("crop values must be finite")
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("crop values must be normalized to 0...1")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("crop dimensions must be positive")
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("crop rectangle exceeds source bounds")


@dataclass(frozen=True, slots=True)
class PanCropKeyframe:
    """A segment-relative crop sample used for smooth target-member pans."""

    time_seconds: float
    crop: NormalizedCrop

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_seconds) or self.time_seconds < 0:
            raise ValueError("crop keyframe time must be non-negative and finite")


@dataclass(frozen=True, slots=True)
class PanCropTrack:
    """Per-shot virtual-camera path.

    FFmpeg's crop x/y expressions are evaluated for every frame.  Zoom is held
    to the widest requested keyframe inside one short edit segment so changes
    in detector box size cannot create visible breathing; scene/beat cuts are
    the safe places where the crop size may change.
    """

    keyframes: tuple[PanCropKeyframe, ...]

    def __post_init__(self) -> None:
        if not self.keyframes:
            raise ValueError("pan crop track needs at least one keyframe")
        if abs(self.keyframes[0].time_seconds) > 1e-9:
            raise ValueError("pan crop track must start at segment time zero")
        previous = -1.0
        ratios: list[float] = []
        for keyframe in self.keyframes:
            if keyframe.time_seconds <= previous:
                raise ValueError("crop keyframe times must be strictly increasing")
            previous = keyframe.time_seconds
            ratios.append(keyframe.crop.width / keyframe.crop.height)
        if max(ratios) - min(ratios) > 0.002:
            raise ValueError("crop keyframes must use one normalized aspect ratio")

    @property
    def fixed_crop_size(self) -> tuple[float, float]:
        ratio = self.keyframes[0].crop.width / self.keyframes[0].crop.height
        height = max(keyframe.crop.height for keyframe in self.keyframes)
        width = ratio * height
        if width > 1.0 + 1e-9 or height > 1.0 + 1e-9:
            raise ValueError("pan crop track cannot fit inside the source frame")
        return (min(width, 1.0), min(height, 1.0))

    def fixed_crop_positions(self) -> tuple[tuple[float, float, float], ...]:
        width, height = self.fixed_crop_size
        positions: list[tuple[float, float, float]] = []
        for keyframe in self.keyframes:
            center_x = keyframe.crop.x + keyframe.crop.width / 2.0
            center_y = keyframe.crop.y + keyframe.crop.height / 2.0
            left = min(max(center_x - width / 2.0, 0.0), 1.0 - width)
            top = min(max(center_y - height / 2.0, 0.0), 1.0 - height)
            positions.append((keyframe.time_seconds, left, top))
        return tuple(positions)

    def simplified(
        self,
        *,
        max_keyframes: int = 80,
        tolerance: float = 0.0015,
    ) -> "PanCropTrack":
        """Reduce a dense tracker path before building an FFmpeg expression.

        FFmpeg's expression parser has a practical nesting limit.  Short edit
        segments rarely reach it, but this Ramer-Douglas-Peucker pass preserves
        bends while placing a hard cap on adversarial/dense paths.  The widest
        keyframe is mandatory so simplification cannot accidentally zoom in.
        """

        if (
            isinstance(max_keyframes, bool)
            or not isinstance(max_keyframes, int)
            or max_keyframes < 3
        ):
            raise ValueError("max_keyframes must be an integer of at least 3")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be positive and finite")
        if len(self.keyframes) <= max_keyframes:
            return self

        positions = self.fixed_crop_positions()
        widest_index = max(
            range(len(self.keyframes)),
            key=lambda index: self.keyframes[index].crop.height,
        )

        def rdp(first: int, last: int, epsilon: float) -> set[int]:
            if last <= first + 1:
                return {first, last}
            start_time, start_x, start_y = positions[first]
            end_time, end_x, end_y = positions[last]
            span = end_time - start_time
            maximum_error = -1.0
            maximum_index = first
            for index in range(first + 1, last):
                time_value, x_value, y_value = positions[index]
                alpha = (time_value - start_time) / span
                expected_x = start_x + (end_x - start_x) * alpha
                expected_y = start_y + (end_y - start_y) * alpha
                error = math.hypot(x_value - expected_x, y_value - expected_y)
                if error > maximum_error:
                    maximum_error = error
                    maximum_index = index
            if maximum_error <= epsilon:
                return {first, last}
            return rdp(first, maximum_index, epsilon) | rdp(
                maximum_index, last, epsilon
            )

        epsilon = tolerance
        mandatory = {0, widest_index, len(self.keyframes) - 1}
        selected: set[int] = set()
        for _ in range(24):
            selected = rdp(0, len(self.keyframes) - 1, epsilon) | mandatory
            if len(selected) <= max_keyframes:
                break
            epsilon *= 1.6
        if len(selected) > max_keyframes:
            optional = sorted(selected - mandatory)
            capacity = max_keyframes - len(mandatory)
            if capacity > 0:
                chosen = {
                    optional[round(index * (len(optional) - 1) / max(1, capacity - 1))]
                    for index in range(capacity)
                }
            else:
                chosen = set()
            selected = mandatory | chosen
        return PanCropTrack(tuple(self.keyframes[index] for index in sorted(selected)))


EDITORIAL_TRANSFORM_EDGE_MARGIN = 0.003


def minimum_roll_safe_zoom(roll_degrees: float) -> float:
    """Return the centered zoom needed to hide 16:9 rotation corners.

    The same bound applies to 9:16 output because the limiting aspect ratio is
    the longer edge divided by the shorter edge.  A small numerical margin is
    included so chroma resampling cannot reveal a one-pixel black wedge.
    """

    if not math.isfinite(roll_degrees):
        raise ValueError("roll must be finite")
    radians = math.radians(abs(roll_degrees))
    long_to_short = 16.0 / 9.0
    base = math.cos(radians) + long_to_short * math.sin(radians)
    strength = min(1.0, abs(roll_degrees))
    # FFmpeg's rotate and zoompan filters use different chroma/interpolation
    # footprints.  Keep roughly two percent extra overscan beyond the analytic
    # luma bound so compressed 4:2:0 edges cannot leak through at low as well
    # as production resolutions.
    margin = EDITORIAL_TRANSFORM_EDGE_MARGIN * strength
    return base / (1.0 - 2.0 * margin) + 0.020 * strength


def _rotated_crop_is_inside_frame(
    *,
    zoom: float,
    center_x: float,
    center_y: float,
    roll_degrees: float,
) -> bool:
    """Check all virtual-camera corners in the unrotated source frame."""

    if abs(roll_degrees) <= 1e-12:
        half = 0.5 / zoom
        return (
            center_x - half >= -1e-9
            and center_x + half <= 1.0 + 1e-9
            and center_y - half >= -1e-9
            and center_y + half <= 1.0 + 1e-9
        )
    angle = math.radians(roll_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    half = 0.5 / zoom
    limit = 0.5 - EDITORIAL_TRANSFORM_EDGE_MARGIN * min(
        1.0, abs(roll_degrees)
    )
    # The track is output-aspect agnostic so it can be shared by landscape and
    # portrait/member modes.  Require the crop to fit under both reciprocal
    # aspect transforms; RenderPlan can then serialize one track safely for
    # either orientation.
    for aspect in (16.0 / 9.0, 9.0 / 16.0):
        for x_value in (center_x - half, center_x + half):
            for y_value in (center_y - half, center_y + half):
                dx = x_value - 0.5
                dy = y_value - 0.5
                source_dx = cosine * dx + sine * dy / aspect
                source_dy = -sine * aspect * dx + cosine * dy
                if (
                    abs(source_dx) > limit + 1e-9
                    or abs(source_dy) > limit + 1e-9
                ):
                    return False
    return True


@dataclass(frozen=True, slots=True)
class EditorialTransformKeyframe:
    """One segment-relative sample for the shared editorial camera layer."""

    time_seconds: float
    zoom: float
    center_x: float = 0.5
    center_y: float = 0.5
    roll_degrees: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.time_seconds,
            self.zoom,
            self.center_x,
            self.center_y,
            self.roll_degrees,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("group transform values must be finite")
        if self.time_seconds < 0.0:
            raise ValueError("group transform time must be non-negative")
        if not 1.0 <= self.zoom <= 1.18:
            raise ValueError("editorial transform zoom must be in [1.0, 1.18]")
        if not 0.0 <= self.center_x <= 1.0 or not 0.0 <= self.center_y <= 1.0:
            raise ValueError("editorial transform center must be normalized")
        if math.hypot(self.center_x - 0.5, self.center_y - 0.5) > 0.040 + 1e-9:
            raise ValueError("editorial transform pan exceeds the four-percent cap")
        if abs(self.roll_degrees) > 1.60:
            raise ValueError("editorial transform roll exceeds the subtle-roll bound")
        if abs(self.roll_degrees) > 1e-9 and (
            self.zoom + 1e-9 < minimum_roll_safe_zoom(self.roll_degrees)
        ):
            raise ValueError("group transform zoom cannot hide transformed frame edges")
        if not _rotated_crop_is_inside_frame(
            zoom=self.zoom,
            center_x=self.center_x,
            center_y=self.center_y,
            roll_degrees=self.roll_degrees,
        ):
            raise ValueError("group transform cannot hide transformed frame edges")


@dataclass(frozen=True, slots=True)
class EditorialTransformTrack:
    """Authored, duration-neutral camera motion shared by both edit modes.

    Unlike ``PanCropTrack``, this track deliberately permits a bounded zoom
    ramp.  It is kept as a separate type so member tracking retains its fixed
    crop-size anti-breathing guarantee.
    """

    keyframes: tuple[EditorialTransformKeyframe, ...]
    reason: str = "subject_scale_matched_hard_cut"

    def __post_init__(self) -> None:
        if not self.keyframes:
            raise ValueError("group transform track needs at least one keyframe")
        if abs(self.keyframes[0].time_seconds) > 1e-9:
            raise ValueError("group transform track must start at segment time zero")
        previous = -1.0
        for keyframe in self.keyframes:
            if keyframe.time_seconds <= previous:
                raise ValueError("group transform keyframe times must be strictly increasing")
            previous = keyframe.time_seconds
        for left, right in zip(self.keyframes, self.keyframes[1:]):
            for step in range(1, 16):
                alpha = step / 16.0
                zoom = left.zoom + (right.zoom - left.zoom) * alpha
                roll_degrees = (
                    left.roll_degrees
                    + (right.roll_degrees - left.roll_degrees) * alpha
                )
                if abs(roll_degrees) > 1e-9 and (
                    zoom + 1e-9 < minimum_roll_safe_zoom(roll_degrees)
                ):
                    raise ValueError(
                        "group transform interpolation reveals transformed frame edges"
                    )
                if not _rotated_crop_is_inside_frame(
                    zoom=zoom,
                    center_x=(
                        left.center_x
                        + (right.center_x - left.center_x) * alpha
                    ),
                    center_y=(
                        left.center_y
                        + (right.center_y - left.center_y) * alpha
                    ),
                    roll_degrees=roll_degrees,
                ):
                    raise ValueError(
                        "group transform interpolation reveals transformed frame edges"
                    )
        if not self.reason.strip():
            raise ValueError("group transform reason cannot be blank")

    @property
    def final_keyframe(self) -> EditorialTransformKeyframe:
        return self.keyframes[-1]

    def evaluate(self, time_seconds: float) -> EditorialTransformKeyframe:
        """Evaluate the exact linear state used by FFmpeg at one frame time."""

        if not math.isfinite(time_seconds) or time_seconds < 0.0:
            raise ValueError("editorial transform evaluation time is invalid")
        if time_seconds <= self.keyframes[0].time_seconds + 1e-12:
            item = self.keyframes[0]
            return EditorialTransformKeyframe(
                time_seconds,
                item.zoom,
                item.center_x,
                item.center_y,
                item.roll_degrees,
            )
        for left, right in zip(self.keyframes, self.keyframes[1:]):
            if time_seconds <= right.time_seconds + 1e-12:
                # Segment-local keyframe times are commonly produced by
                # subtracting two independently quantized global timestamps.
                # At an exact output-frame boundary that can leave the query a
                # few ulps beyond ``right`` while still inside the tolerance
                # above.  Never extrapolate a bounded transform merely because
                # the tolerant endpoint comparison admitted that timestamp.
                alpha = min(
                    1.0,
                    max(
                        0.0,
                        (time_seconds - left.time_seconds)
                        / (right.time_seconds - left.time_seconds),
                    ),
                )
                return EditorialTransformKeyframe(
                    time_seconds,
                    left.zoom + (right.zoom - left.zoom) * alpha,
                    left.center_x + (right.center_x - left.center_x) * alpha,
                    left.center_y + (right.center_y - left.center_y) * alpha,
                    left.roll_degrees
                    + (right.roll_degrees - left.roll_degrees) * alpha,
                )
        item = self.keyframes[-1]
        return EditorialTransformKeyframe(
            time_seconds,
            item.zoom,
            item.center_x,
            item.center_y,
            item.roll_degrees,
        )


def _editorial_state_values(
    track: EditorialTransformTrack | None,
    time_seconds: float,
) -> tuple[float, float, float, float]:
    if track is None:
        return (1.0, 0.5, 0.5, 0.0)
    item = track.evaluate(time_seconds)
    return (item.zoom, item.center_x, item.center_y, item.roll_degrees)


def _tier_d_transform_is_frame_continuous(
    left: EditorialTransformTrack | None,
    right: EditorialTransformTrack | None,
    *,
    left_frame_count: int,
    right_frame_count: int,
    fps: float,
) -> bool:
    """Reject a virtual-camera reset hidden inside one continuous source shot."""

    frame = 1.0 / fps
    left_last_time = (left_frame_count - 1) * frame
    left_previous_time = max(0.0, left_last_time - frame)
    right_next_time = frame if right_frame_count > 1 else 0.0
    before = _editorial_state_values(left, left_previous_time)
    outgoing = _editorial_state_values(left, left_last_time)
    incoming = _editorial_state_values(right, 0.0)
    after = _editorial_state_values(right, right_next_time)
    maximum_steps = (
        0.18 / (0.55 * fps),
        0.04 / (0.55 * fps),
        0.04 / (0.55 * fps),
        1.60 / (0.55 * fps),
    )
    for index, maximum_step in enumerate(maximum_steps):
        previous_step = outgoing[index] - before[index]
        cross_step = incoming[index] - outgoing[index]
        next_step = after[index] - incoming[index]
        if abs(cross_step) > maximum_step + 1e-8:
            return False
        # A true global piecewise-linear curve has the cross-boundary step on
        # either the preceding or following slope when a keypoint lands exactly
        # at the split.  This catches a fractional-frame reset/velocity pulse
        # while allowing an intentional slope change at that keypoint.
        closest_neighbor_error = min(
            abs(cross_step - previous_step),
            abs(cross_step - next_step),
        )
        reference_step = max(
            abs(cross_step),
            abs(previous_step),
            abs(next_step),
        )
        if closest_neighbor_error > max(1e-5, 0.25 * reference_step):
            return False
    return True


@dataclass(frozen=True, slots=True)
class SourceLookAdjustment:
    """Small, duration-neutral source look correction consumed by FFmpeg.

    The bounds are intentionally much tighter than FFmpeg's legal ranges.  A
    source-quality gate, rather than this color layer, remains responsible for
    rejecting soft, clipped, or low-resolution footage.
    """

    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    temperature_balance: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.brightness,
            self.contrast,
            self.saturation,
            self.temperature_balance,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("source look adjustment values must be finite")
        if not -0.05 <= self.brightness <= 0.05:
            raise ValueError("source look brightness exceeds the conservative bound")
        if not 0.94 <= self.contrast <= 1.06:
            raise ValueError("source look contrast exceeds the conservative bound")
        if not 0.94 <= self.saturation <= 1.06:
            raise ValueError("source look saturation exceeds the conservative bound")
        if not -0.03 <= self.temperature_balance <= 0.03:
            raise ValueError(
                "source look temperature balance exceeds the conservative bound"
            )

    @property
    def is_identity(self) -> bool:
        return (
            abs(self.brightness) <= 1e-12
            and abs(self.contrast - 1.0) <= 1e-12
            and abs(self.saturation - 1.0) <= 1e-12
            and abs(self.temperature_balance) <= 1e-12
        )


@dataclass(frozen=True, slots=True)
class BoundaryLookKeyframe:
    """One segment-local residual look sample.

    This layer is deliberately separate from :class:`SourceLookAdjustment`.
    The latter is the once-per-source static normalization; this keyframe is a
    short, post-normalization residual used only next to a reviewed hard cut.
    Keeping the two types separate makes accidental double application visible
    in both the render plan and the FFmpeg graph.
    """

    time_seconds: float
    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    temperature_balance: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_seconds) or self.time_seconds < 0.0:
            raise ValueError("boundary look keyframe time must be non-negative")
        # Reuse the renderer's conservative numeric contract.  Boundary-local
        # analysis applies tighter configuration bounds before constructing a
        # keyframe; this check is the final serialization safety net.
        SourceLookAdjustment(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            temperature_balance=self.temperature_balance,
        )

    @property
    def is_identity(self) -> bool:
        return (
            abs(self.brightness) <= 1e-12
            and abs(self.contrast - 1.0) <= 1e-12
            and abs(self.saturation - 1.0) <= 1e-12
            and abs(self.temperature_balance) <= 1e-12
        )


@dataclass(frozen=True, slots=True)
class BoundaryLookTrack:
    """A bounded per-frame look ramp around cross-source hard cuts."""

    keyframes: tuple[BoundaryLookKeyframe, ...]
    reason: str = "boundary_local_post_global_look_residual"

    def __post_init__(self) -> None:
        if not self.keyframes:
            raise ValueError("boundary look track needs at least one keyframe")
        if abs(self.keyframes[0].time_seconds) > 1e-9:
            raise ValueError("boundary look track must start at segment time zero")
        previous = -1.0
        for keyframe in self.keyframes:
            if keyframe.time_seconds <= previous:
                raise ValueError(
                    "boundary look keyframe times must be strictly increasing"
                )
            previous = keyframe.time_seconds
        if not self.reason.strip():
            raise ValueError("boundary look track reason cannot be blank")

    @property
    def is_identity(self) -> bool:
        return all(keyframe.is_identity for keyframe in self.keyframes)

    @property
    def first_keyframe(self) -> BoundaryLookKeyframe:
        return self.keyframes[0]

    @property
    def final_keyframe(self) -> BoundaryLookKeyframe:
        return self.keyframes[-1]

    def evaluate(self, time_seconds: float) -> BoundaryLookKeyframe:
        if not math.isfinite(time_seconds) or time_seconds < 0.0:
            raise ValueError("boundary look evaluation time is invalid")
        if time_seconds <= self.keyframes[0].time_seconds + 1e-12:
            item = self.keyframes[0]
            return BoundaryLookKeyframe(
                time_seconds,
                item.brightness,
                item.contrast,
                item.saturation,
                item.temperature_balance,
            )
        for left, right in zip(self.keyframes, self.keyframes[1:]):
            if time_seconds <= right.time_seconds + 1e-12:
                alpha = min(
                    1.0,
                    max(
                        0.0,
                        (time_seconds - left.time_seconds)
                        / (right.time_seconds - left.time_seconds),
                    ),
                )
                return BoundaryLookKeyframe(
                    time_seconds,
                    left.brightness
                    + (right.brightness - left.brightness) * alpha,
                    left.contrast + (right.contrast - left.contrast) * alpha,
                    left.saturation
                    + (right.saturation - left.saturation) * alpha,
                    left.temperature_balance
                    + (
                        right.temperature_balance
                        - left.temperature_balance
                    )
                    * alpha,
                )
        item = self.keyframes[-1]
        return BoundaryLookKeyframe(
            time_seconds,
            item.brightness,
            item.contrast,
            item.saturation,
            item.temperature_balance,
        )


@dataclass(frozen=True, slots=True)
class RenderSegment:
    source_index: int
    source_start: float
    duration: float
    master_start: float
    crop: NormalizedCrop | None = None
    pan_crop: PanCropTrack | None = None
    editorial_transform: EditorialTransformTrack | None = None
    # Canonical ``duration`` controls the output timeline.  A local time-warp
    # can require a slightly longer or shorter interval from the source clock;
    # the filter retimes that interval back to the canonical duration.
    source_duration: float | None = None
    # Optional static, evidence-gated color correction.  It changes pixels but
    # never cadence, source clocks, geometry, or output duration.
    look_adjustment: SourceLookAdjustment | None = None
    # Optional boundary-local residual measured *after* ``look_adjustment``.
    # It is expressed as a short smooth ramp and is never used as a second
    # source-global grade.
    boundary_look_track: BoundaryLookTrack | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
            or self.source_index < 0
        ):
            raise ValueError("source_index must be non-negative")
        if not all(
            math.isfinite(value)
            for value in (self.source_start, self.duration, self.master_start)
        ):
            raise ValueError("segment times must be finite")
        if self.source_start < 0 or self.master_start < 0:
            raise ValueError("segment times must be non-negative")
        if self.duration <= 0:
            raise ValueError("segment duration must be positive")
        if self.source_duration is not None:
            if not math.isfinite(self.source_duration) or self.source_duration <= 0:
                raise ValueError("source_duration must be positive and finite")
            if (
                abs(self.source_duration / self.duration - 1.0)
                > MAXIMUM_RENDER_TIMEWARP_RATE_DELTA + 1e-9
            ):
                raise ValueError("local source rate exceeds the render safety bound")
        if self.look_adjustment is not None and not isinstance(
            self.look_adjustment, SourceLookAdjustment
        ):
            raise TypeError("look_adjustment must be a SourceLookAdjustment")
        if self.boundary_look_track is not None and not isinstance(
            self.boundary_look_track, BoundaryLookTrack
        ):
            raise TypeError("boundary_look_track must be a BoundaryLookTrack")
        if (
            self.boundary_look_track is not None
            and self.boundary_look_track.keyframes[-1].time_seconds
            > self.duration + 1e-6
        ):
            raise ValueError("boundary look keyframes must stay inside the segment")
        if self.crop is not None and self.pan_crop is not None:
            raise ValueError("segment cannot use both static and panning crops")
        if (
            self.pan_crop is not None
            and self.pan_crop.keyframes[-1].time_seconds > self.duration + 1e-6
        ):
            raise ValueError("crop keyframes must stay inside the segment")
        if (
            self.editorial_transform is not None
            and self.editorial_transform.keyframes[-1].time_seconds
            > self.duration + 1e-6
        ):
            raise ValueError("group transform keyframes must stay inside the segment")

    @property
    def effective_source_duration(self) -> float:
        return self.duration if self.source_duration is None else self.source_duration

    @property
    def source_playback_rate(self) -> float:
        return self.effective_source_duration / self.duration

    @property
    def source_end(self) -> float:
        """Exact source-local endpoint consumed by this rendered segment."""

        return self.source_start + self.effective_source_duration


def _base_crop_values(
    segment: RenderSegment,
    time_seconds: float,
) -> tuple[float, float, float, float]:
    if segment.crop is not None:
        return (
            segment.crop.x,
            segment.crop.y,
            segment.crop.width,
            segment.crop.height,
        )
    if segment.pan_crop is None:
        return (0.0, 0.0, 1.0, 1.0)
    width, height = segment.pan_crop.fixed_crop_size
    positions = segment.pan_crop.fixed_crop_positions()
    if time_seconds <= positions[0][0] + 1e-9:
        return (positions[0][1], positions[0][2], width, height)
    for left, right in zip(positions, positions[1:]):
        if time_seconds <= right[0] + 1e-9:
            span = right[0] - left[0]
            if span <= 1e-9:
                return (right[1], right[2], width, height)
            alpha = (time_seconds - left[0]) / span
            return (
                left[1] + (right[1] - left[1]) * alpha,
                left[2] + (right[2] - left[2]) * alpha,
                width,
                height,
            )
    return (positions[-1][1], positions[-1][2], width, height)


def _tier_d_base_crop_is_frame_continuous(
    left: RenderSegment,
    right: RenderSegment,
    *,
    left_frame_count: int,
    right_frame_count: int,
    fps: float,
) -> bool:
    frame = 1.0 / fps
    left_last_time = min(left.duration, (left_frame_count - 1) * frame)
    left_previous_time = max(0.0, left_last_time - frame)
    right_next_time = min(right.duration, frame if right_frame_count > 1 else 0.0)
    before = _base_crop_values(left, left_previous_time)
    outgoing = _base_crop_values(left, left_last_time)
    incoming = _base_crop_values(right, 0.0)
    after = _base_crop_values(right, right_next_time)
    maximum_steps = (
        0.04 / (0.55 * fps),
        0.04 / (0.55 * fps),
        0.18 / (0.55 * fps),
        0.18 / (0.55 * fps),
    )
    for index, maximum_step in enumerate(maximum_steps):
        previous_step = outgoing[index] - before[index]
        cross_step = incoming[index] - outgoing[index]
        next_step = after[index] - incoming[index]
        if abs(cross_step) > maximum_step + 1e-8:
            return False
        closest_neighbor_error = min(
            abs(cross_step - previous_step),
            abs(cross_step - next_step),
        )
        reference_step = max(
            abs(cross_step),
            abs(previous_step),
            abs(next_step),
        )
        if closest_neighbor_error > max(1e-5, 0.25 * reference_step):
            return False
    return True


class TransitionKind(str, Enum):
    """Video transition applied at one nominal segment boundary."""

    HARD_CUT = "hard_cut"
    FACE_DISSOLVE = "face_dissolve"


FACE_DISSOLVE_ALPHA_CURVE = "normalized_logistic_sigmoid_k12_v1"
# FFmpeg xfade exposes P=1 at the first transition frame and P=0 at the last.
# Normalize logistic(k=12) so the first/last pictures remain exact rather than
# retaining a faint ghost at either endpoint.  This follows the non-linear
# ease-in/out finding in AutoStageMix while preserving our duration-neutral,
# canonical-time overlap contract.
_FACE_DISSOLVE_LOGISTIC_ENDPOINT = 0.0024726231566347743
_FACE_DISSOLVE_LOGISTIC_SPAN = 0.9950547536867305
_FACE_DISSOLVE_A_WEIGHT_EXPRESSION = (
    "((1/(1+exp(-12*(P-0.5)))-"
    f"{_FACE_DISSOLVE_LOGISTIC_ENDPOINT:.19f})/"
    f"{_FACE_DISSOLVE_LOGISTIC_SPAN:.16f})"
)
_FACE_DISSOLVE_XFADE_EXPRESSION = (
    f"A*{_FACE_DISSOLVE_A_WEIGHT_EXPRESSION}+"
    f"B*(1-{_FACE_DISSOLVE_A_WEIGHT_EXPRESSION})"
)


@dataclass(frozen=True, slots=True)
class RenderTransition:
    """Explicit per-boundary transition metadata.

    ``FACE_DISSOLVE`` uses an outgoing-only source handle.  The incoming shot
    still begins at the nominal cut, so both pictures show the same canonical
    song time during the overlap and the edit's total frame count is unchanged.
    """

    kind: TransitionKind = TransitionKind.HARD_CUT
    frame_count: int = 0
    reason: str = "default_hard_cut"
    face_mismatch: float | None = None
    tier: str | None = None
    appearance_min_cosine_similarity: float | None = None
    future_pose_max_mismatch: float | None = None
    future_silhouette_max_mismatch: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TransitionKind):
            object.__setattr__(self, "kind", TransitionKind(self.kind))
        if (
            isinstance(self.frame_count, bool)
            or not isinstance(self.frame_count, int)
            or self.frame_count < 0
        ):
            raise ValueError("transition frame_count must be a non-negative integer")
        if self.kind is TransitionKind.HARD_CUT and self.frame_count != 0:
            raise ValueError("hard cuts cannot have overlap frames")
        if self.kind is TransitionKind.FACE_DISSOLVE and self.frame_count < 3:
            raise ValueError("face dissolves need at least three frames")
        if not self.reason:
            raise ValueError("transition reason cannot be blank")
        if self.tier is not None and self.tier not in {"A", "B", "C", "D"}:
            raise ValueError("transition tier must be one of A, B, C, or D")
        if self.face_mismatch is not None and (
            not math.isfinite(self.face_mismatch)
            or not 0.0 <= self.face_mismatch <= 1.0
        ):
            raise ValueError("face mismatch must be normalized")
        if self.appearance_min_cosine_similarity is not None and (
            not math.isfinite(self.appearance_min_cosine_similarity)
            or not -1.0 <= self.appearance_min_cosine_similarity <= 1.0
        ):
            raise ValueError("appearance cosine similarity must be normalized")
        for value in (
            self.future_pose_max_mismatch,
            self.future_silhouette_max_mismatch,
        ):
            if value is not None and (
                not math.isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError("future body mismatch must be normalized")

    def duration_seconds(self, fps: float) -> float:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        return self.frame_count / fps

    def to_dict(self, fps: float) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "frame_count": self.frame_count,
            "duration_seconds": self.duration_seconds(fps),
            "reason": self.reason,
            "face_mismatch": self.face_mismatch,
            "tier": self.tier,
            "appearance_min_cosine_similarity": (
                self.appearance_min_cosine_similarity
            ),
            "future_pose_max_mismatch": self.future_pose_max_mismatch,
            "future_silhouette_max_mismatch": (
                self.future_silhouette_max_mismatch
            ),
            "alpha_curve": (
                FACE_DISSOLVE_ALPHA_CURVE
                if self.kind is TransitionKind.FACE_DISSOLVE
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class RenderPlan:
    sources: tuple[Path, ...]
    segments: tuple[RenderSegment, ...]
    reference_source: int
    aspect_ratio: AspectRatio
    output: Path
    master_audio_start: float = 0.0
    width: int = 1920
    height: int = 1080
    fps: float = 29.97
    transitions: tuple[RenderTransition, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.aspect_ratio, AspectRatio):
            object.__setattr__(self, "aspect_ratio", AspectRatio(self.aspect_ratio))
        if len(self.sources) < 1:
            raise ValueError("render plan needs at least one source")
        if not self.segments:
            raise ValueError("render plan needs at least one segment")
        if (
            isinstance(self.reference_source, bool)
            or not isinstance(self.reference_source, int)
            or not 0 <= self.reference_source < len(self.sources)
        ):
            raise ValueError("reference_source is outside the source list")
        if not math.isfinite(self.master_audio_start) or self.master_audio_start < 0:
            raise ValueError("master_audio_start must be non-negative")
        if (
            isinstance(self.width, bool)
            or not isinstance(self.width, int)
            or isinstance(self.height, bool)
            or not isinstance(self.height, int)
            or self.width <= 0
            or self.height <= 0
            or self.width % 2
            or self.height % 2
        ):
            raise ValueError("output dimensions must be positive even integers")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive and finite")
        expected_ratio = 16 / 9 if self.aspect_ratio is AspectRatio.LANDSCAPE else 9 / 16
        if abs(self.width / self.height - expected_ratio) > 0.002:
            raise ValueError("output dimensions do not match aspect_ratio")
        if any(segment.source_index >= len(self.sources) for segment in self.segments):
            raise ValueError("segment references an unknown source")
        resolved_sources = {source.resolve() for source in self.sources}
        if self.output.resolve() in resolved_sources:
            raise ValueError("output must not overwrite a source video")
        expected = 0.0
        for segment in self.segments:
            if abs(segment.master_start - expected) > 1e-6:
                raise ValueError("segments must be contiguous and ordered on the master timeline")
            expected += segment.duration
        if not self.transitions and len(self.segments) > 1:
            object.__setattr__(
                self,
                "transitions",
                tuple(RenderTransition() for _ in range(len(self.segments) - 1)),
            )
        if len(self.transitions) != len(self.segments) - 1:
            raise ValueError("render plan needs one transition per segment boundary")
        counts = segment_frame_counts(self)
        for segment, frame_count in zip(self.segments, counts, strict=True):
            if segment.editorial_transform is None:
                continue
            for frame_index in range(frame_count):
                try:
                    segment.editorial_transform.evaluate(frame_index / self.fps)
                except ValueError as exc:
                    raise ValueError(
                        "editorial transform is unsafe at an output-frame timestamp"
                    ) from exc
        for index, transition in enumerate(self.transitions):
            left_boundary_look = self.segments[index].boundary_look_track
            right_boundary_look = self.segments[index + 1].boundary_look_track
            left_endpoint_identity = (
                left_boundary_look is None
                or left_boundary_look.final_keyframe.is_identity
            )
            right_endpoint_identity = (
                right_boundary_look is None
                or right_boundary_look.first_keyframe.is_identity
            )
            if (
                transition.tier == "D"
                and self.segments[index].source_index
                != self.segments[index + 1].source_index
            ):
                raise ValueError("tier D transition requires the same source")
            if transition.tier == "D" and abs(
                self.segments[index].source_end
                - self.segments[index + 1].source_start
            ) > 1e-4:
                raise ValueError("tier D transition requires a contiguous source clock")
            if transition.tier == "D" and not _tier_d_base_crop_is_frame_continuous(
                self.segments[index],
                self.segments[index + 1],
                left_frame_count=counts[index],
                right_frame_count=counts[index + 1],
                fps=self.fps,
            ):
                raise ValueError("same-source hold cannot reset its base crop")
            if transition.tier == "D" and not _tier_d_transform_is_frame_continuous(
                self.segments[index].editorial_transform,
                self.segments[index + 1].editorial_transform,
                left_frame_count=counts[index],
                right_frame_count=counts[index + 1],
                fps=self.fps,
            ):
                raise ValueError(
                    "same-source hold cannot reset editorial camera motion"
                )
            if transition.tier == "D" and not (
                left_endpoint_identity and right_endpoint_identity
            ):
                raise ValueError(
                    "same-source hold cannot contain a boundary-local look reset"
                )
            if (
                transition.kind is TransitionKind.FACE_DISSOLVE
                and (
                    self.segments[index].editorial_transform is not None
                    or self.segments[index + 1].editorial_transform is not None
                )
            ):
                raise ValueError(
                    "face dissolve cannot use an unreviewed editorial transform"
                )
            if transition.kind is TransitionKind.FACE_DISSOLVE and not (
                left_endpoint_identity and right_endpoint_identity
            ):
                raise ValueError(
                    "face dissolve cannot use a hard-cut boundary look ramp"
                )
            if (
                transition.kind is TransitionKind.FACE_DISSOLVE
                and transition.frame_count > counts[index + 1]
            ):
                raise ValueError("face dissolve cannot exceed the incoming segment")
            if (
                transition.kind is TransitionKind.FACE_DISSOLVE
                and self.segments[index].source_index
                == self.segments[index + 1].source_index
            ):
                raise ValueError("face dissolve requires a real source change")

    @property
    def duration(self) -> float:
        return sum(segment.duration for segment in self.segments)

    @property
    def output_frame_count(self) -> int:
        return int(round(self.duration * self.fps))

    @property
    def render_duration(self) -> float:
        return self.output_frame_count / self.fps

    def source_usage_payload(self) -> dict[str, object]:
        """Return actual post-selection usage for every render-plan source."""

        segment_counts = [0] * len(self.sources)
        durations = [0.0] * len(self.sources)
        for segment in self.segments:
            segment_counts[segment.source_index] += 1
            durations[segment.source_index] += segment.duration
        used = tuple(
            index for index, count in enumerate(segment_counts) if count > 0
        )
        return {
            "index_domain": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
            "distinct_source_count": len(used),
            "used_source_indices": list(used),
            "total_segment_count": len(self.segments),
            "total_duration_seconds": self.duration,
            "sources": [
                {
                    "source_index": index,
                    "segment_count": segment_counts[index],
                    "duration_seconds": durations[index],
                }
                for index in range(len(self.sources))
            ],
        }

    def to_json(self) -> dict[str, object]:
        serialized_segments: list[dict[str, object]] = []
        for segment in self.segments:
            payload = asdict(segment)
            payload["effective_source_duration"] = segment.effective_source_duration
            payload["source_playback_rate"] = segment.source_playback_rate
            payload["source_local_start"] = segment.source_start
            payload["source_local_end"] = segment.source_end
            serialized_segments.append(payload)
        return {
            "sources": [str(path) for path in self.sources],
            "segments": serialized_segments,
            "reference_source": self.reference_source,
            "source_index_domains": {
                "sources[]": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
                "reference_source": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
                "segments[].source_index": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
                "source_usage.used_source_indices[]": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
                "source_usage.sources[].source_index": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
            },
            "aspect_ratio": self.aspect_ratio.value,
            "output": str(self.output),
            "master_audio_start": self.master_audio_start,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "transitions": [
                transition.to_dict(self.fps) for transition in self.transitions
            ],
            "duration": self.duration,
            "render_duration": self.render_duration,
            "output_frame_count": self.output_frame_count,
            "source_usage": self.source_usage_payload(),
        }


def output_size(aspect_ratio: AspectRatio, long_edge: int = 1920) -> tuple[int, int]:
    if isinstance(long_edge, bool) or not isinstance(long_edge, int) or long_edge <= 0:
        raise ValueError("long_edge must be a positive integer")
    long_edge -= long_edge % 2
    short_edge = int(round(long_edge * 9 / 16))
    short_edge -= short_edge % 2
    if aspect_ratio is AspectRatio.PORTRAIT:
        return (short_edge, long_edge)
    return (long_edge, short_edge)


def aligned_render_segment(
    *,
    source_index: int,
    reference_start: float,
    common_master_start: float,
    source_offset: float,
    duration: float,
    crop: NormalizedCrop | None = None,
    pan_crop: PanCropTrack | None = None,
    editorial_transform: EditorialTransformTrack | None = None,
    timewarp: TimeWarpMap | None = None,
) -> RenderSegment:
    """Map one canonical reference span into output and source time.

    Without ``timewarp``, alignment uses the invariant
    ``source_time = reference_time + offset``.  A checked time-warp maps both
    endpoints and records the local source span separately; rendering then
    compensates its small rate difference while retaining canonical duration.
    The output starts at zero even when the common usable reference range starts
    later, so ``master_start`` deliberately subtracts ``common_master_start``.
    """

    values = (reference_start, common_master_start, source_offset, duration)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("alignment times must be finite")
    if reference_start < common_master_start:
        raise ValueError("reference_start precedes the common master range")
    if timewarp is not None and not isinstance(timewarp, TimeWarpMap):
        raise TypeError("timewarp must be a TimeWarpMap")
    if timewarp is not None and not math.isclose(
        timewarp.global_offset_seconds,
        source_offset,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("timewarp global offset does not match source_offset")
    if timewarp is None or timewarp.scalar_fallback:
        source_start = reference_start + source_offset
        source_duration: float | None = None
    else:
        source_start, source_end = timewarp.master_span_to_source(
            reference_start, reference_start + duration
        )
        source_duration = source_end - source_start
    return RenderSegment(
        source_index=source_index,
        source_start=source_start,
        duration=duration,
        master_start=reference_start - common_master_start,
        crop=crop,
        pan_crop=pan_crop,
        editorial_transform=editorial_transform,
        source_duration=source_duration,
    )


def _escape_filter_expression(expression: str) -> str:
    return expression.replace("\\", "\\\\").replace("'", "\\'")


def _piecewise_linear_expression(
    positions: tuple[tuple[float, ...], ...],
    coordinate_index: int,
    *,
    time_variable: str = "t",
) -> str:
    """Return an FFmpeg expression interpolating one keyframe component."""

    if coordinate_index < 1 or any(
        coordinate_index >= len(position) for position in positions
    ):
        raise ValueError("coordinate_index is outside the keyframe tuple")
    if not time_variable:
        raise ValueError("time_variable cannot be blank")
    clock = time_variable if time_variable.isidentifier() else f"({time_variable})"
    if len(positions) == 1:
        return f"{positions[0][coordinate_index]:.10f}"
    expression = f"{positions[-1][coordinate_index]:.10f}"
    for left, right in reversed(tuple(zip(positions, positions[1:]))):
        start_time = left[0]
        end_time = right[0]
        start_value = left[coordinate_index]
        end_value = right[coordinate_index]
        slope = (end_value - start_value) / (end_time - start_time)
        linear = (
            f"({start_value:.10f}+({slope:.10f})*"
            f"({clock}-{start_time:.10f}))"
        )
        expression = (
            f"if(lt({clock},{end_time:.10f}),{linear},{expression})"
        )
    return expression


def segment_frame_counts(plan: RenderPlan) -> tuple[int, ...]:
    """Quantize global cut boundaries once so per-cut rounding cannot drift."""

    counts: list[int] = []
    cumulative_time = 0.0
    previous_boundary = 0
    for segment in plan.segments:
        cumulative_time += segment.duration
        boundary = int(round(cumulative_time * plan.fps))
        count = boundary - previous_boundary
        if count < 2:
            raise ValueError(
                f"segment at {segment.master_start:.6f}s is shorter than two output frames"
            )
        counts.append(count)
        previous_boundary = boundary
    if sum(counts) != plan.output_frame_count:
        raise ValueError("quantized segment frames do not match the output duration")
    return tuple(counts)


def _editorial_transform_filters(
    track: EditorialTransformTrack,
    *,
    width: int,
    height: int,
    fps: float,
) -> list[str]:
    """Build a one-frame-in/one-frame-out authored camera transform."""

    samples = tuple(
        (
            keyframe.time_seconds,
            keyframe.zoom,
            keyframe.center_x,
            keyframe.center_y,
            math.radians(keyframe.roll_degrees),
        )
        for keyframe in track.keyframes
    )
    output_clock = f"on/{fps:.10f}"
    zoom_expression = _escape_filter_expression(
        _piecewise_linear_expression(
            samples,
            1,
            time_variable=output_clock,
        )
    )
    center_x_expression = _escape_filter_expression(
        _piecewise_linear_expression(
            samples,
            2,
            time_variable=output_clock,
        )
    )
    center_y_expression = _escape_filter_expression(
        _piecewise_linear_expression(
            samples,
            3,
            time_variable=output_clock,
        )
    )
    filters = [
        f"scale={width}:{height}:force_original_aspect_ratio=increase",
        f"crop={width}:{height}",
        "setsar=1",
        # Canonicalize after local time-warp before zoompan. zoompan emits one
        # output frame per input frame, so the small source-rate correction is
        # retained instead of being silently replaced by source cadence.
        f"fps={fps:.6f}",
    ]
    if any(abs(keyframe.roll_degrees) > 1e-9 for keyframe in track.keyframes):
        roll_expression = _escape_filter_expression(
            _piecewise_linear_expression(samples, 4)
        )
        filters.append(
            f"rotate=angle='{roll_expression}':ow=iw:oh=ih:"
            "fillcolor=black"
        )
    filters.append(
        "zoompan="
        f"z='{zoom_expression}':"
        "x='max(0,min(iw-iw/zoom,"
        f"iw*({center_x_expression})-iw/(2*zoom)))':"
        "y='max(0,min(ih-ih/zoom,"
        f"ih*({center_y_expression})-ih/(2*zoom)))':"
        f"d=1:s={width}x{height}:fps={fps:.6f}"
    )
    return filters


def _source_look_filters(
    adjustment: SourceLookAdjustment | None,
) -> list[str]:
    """Translate a bounded look adjustment into duration-neutral filters."""

    if adjustment is None or adjustment.is_identity:
        return []
    filters: list[str] = []
    if (
        abs(adjustment.brightness) > 1e-12
        or abs(adjustment.contrast - 1.0) > 1e-12
        or abs(adjustment.saturation - 1.0) > 1e-12
    ):
        filters.append(
            "eq="
            f"brightness={adjustment.brightness:.6f}:"
            f"contrast={adjustment.contrast:.6f}:"
            f"saturation={adjustment.saturation:.6f}"
        )
    if abs(adjustment.temperature_balance) > 1e-12:
        warm = adjustment.temperature_balance
        cool = -warm
        # Apply the same subtle red/blue shift to shadows, midtones, and
        # highlights.  preserve-lightness prevents this temperature repair
        # from undoing the separately bounded luma match above.
        filters.append(
            "colorbalance="
            f"rs={warm:.6f}:rm={warm:.6f}:rh={warm:.6f}:"
            f"bs={cool:.6f}:bm={cool:.6f}:bh={cool:.6f}:pl=1"
        )
    return filters


def _boundary_look_filters(
    track: BoundaryLookTrack | None,
) -> list[str]:
    """Translate one post-global boundary ramp into frame-evaluated filters.

    Luma/contrast/saturation remain on FFmpeg's inexpensive ``eq`` path.  A
    separately bounded temperature term uses a luma-preserving chroma-affine
    ``geq`` ramp in the native YUV domain; it is continuous in output time and
    is admitted only after the matched production-render proxy proves no
    sharpness, structure, exposure, or color regression.  Unlike an RGB
    round-trip, an identity chroma expression is sample-exact, so the neutral
    interior of a repaired shot cannot pump detail or alter its next boundary.
    """

    if track is None or track.is_identity:
        return []
    samples = tuple(
        (
            keyframe.time_seconds,
            keyframe.brightness,
            keyframe.contrast,
            keyframe.saturation,
            keyframe.temperature_balance,
        )
        for keyframe in track.keyframes
    )
    brightness = _escape_filter_expression(
        _piecewise_linear_expression(samples, 1)
    )
    contrast = _escape_filter_expression(
        _piecewise_linear_expression(samples, 2)
    )
    saturation = _escape_filter_expression(
        _piecewise_linear_expression(samples, 3)
    )
    filters = [
        "eq="
        f"brightness='{brightness}':"
        f"contrast='{contrast}':"
        f"saturation='{saturation}':eval=frame"
    ]
    if any(
        abs(keyframe.temperature_balance) > 1e-12
        for keyframe in track.keyframes
    ):
        temperature = _escape_filter_expression(
            _piecewise_linear_expression(
                samples,
                4,
                time_variable="T",
            )
        )
        # 128 code values per unit balance keeps the maximum +/-0.020 ramp to
        # a +/-2.56-code chroma move.  Luma is copied exactly; positive balance
        # moves Cr upward and Cb downward (warmer), with the inverse for cool.
        # At balance zero all three planes are copied sample-for-sample.
        filters.append(
            "geq="
            "lum='lum(X,Y)':"
            f"cb='clip(cb(X,Y)-128*({temperature}),0,255)':"
            f"cr='clip(cr(X,Y)+128*({temperature}),0,255)'"
        )
    return filters


def _video_chain(
    *,
    source_index: int,
    source_start: float,
    source_duration: float,
    timeline_duration: float,
    crop: NormalizedCrop | None,
    pan_crop: PanCropTrack | None,
    editorial_transform: EditorialTransformTrack | None,
    look_adjustment: SourceLookAdjustment | None,
    boundary_look_track: BoundaryLookTrack | None,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    output_label: str,
) -> str:
    playback_pts_scale = timeline_duration / source_duration
    setpts = (
        "setpts=PTS-STARTPTS"
        if math.isclose(playback_pts_scale, 1.0, rel_tol=0.0, abs_tol=1e-9)
        else f"setpts={playback_pts_scale:.12f}*(PTS-STARTPTS)"
    )
    chain = [
        f"[{source_index}:v]trim=start={source_start:.6f}:duration={source_duration:.6f}",
        setpts,
    ]
    # Normalize the source look before any crop or scale so the adjustment is
    # independent of output geometry and is also shared by dissolve handles.
    chain.extend(_source_look_filters(look_adjustment))
    # Boundary-local evidence is measured in the already-normalized domain.
    # Applying this second, explicitly residual layer after the static source
    # grade prevents the source-global correction from being counted twice.
    chain.extend(_boundary_look_filters(boundary_look_track))
    if crop is not None:
        chain.append(
            "crop="
            f"w=trunc(iw*{crop.width:.8f}/2)*2:"
            f"h=trunc(ih*{crop.height:.8f}/2)*2:"
            f"x=iw*{crop.x:.8f}:y=ih*{crop.y:.8f}"
        )
    elif pan_crop is not None:
        pan_crop = pan_crop.simplified()
        width_ratio, height_ratio = pan_crop.fixed_crop_size
        positions = pan_crop.fixed_crop_positions()
        x_expression = _escape_filter_expression(
            _piecewise_linear_expression(positions, 1)
        )
        y_expression = _escape_filter_expression(
            _piecewise_linear_expression(positions, 2)
        )
        chain.append(
            "crop="
            f"w=trunc(iw*{width_ratio:.8f}/2)*2:"
            f"h=trunc(ih*{height_ratio:.8f}/2)*2:"
            f"x='iw*({x_expression})':y='ih*({y_expression})'"
        )
    if editorial_transform is not None:
        chain.extend(
            _editorial_transform_filters(
                editorial_transform,
                width=width,
                height=height,
                fps=fps,
            )
        )
    else:
        chain.extend(
            [
                f"scale={width}:{height}:force_original_aspect_ratio=decrease",
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
                "setsar=1",
                f"fps={fps:.6f}",
            ]
        )
    chain.extend(
        [
            "tpad=stop_mode=clone:stop_duration=1",
            f"trim=end_frame={frame_count}",
            f"setpts=N/({fps:.6f}*TB)",
            "settb=AVTB",
            "format=yuv420p",
        ]
    )
    return ",".join(chain) + f"[{output_label}]"


def _video_filter(
    index: int,
    segment: RenderSegment,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    extension_frames: int = 0,
) -> str:
    if extension_frames == 0:
        return _video_chain(
            source_index=segment.source_index,
            source_start=segment.source_start,
            source_duration=segment.effective_source_duration,
            timeline_duration=segment.duration,
            crop=segment.crop,
            pan_crop=segment.pan_crop,
            editorial_transform=segment.editorial_transform,
            look_adjustment=segment.look_adjustment,
            boundary_look_track=segment.boundary_look_track,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            output_label=f"v{index}",
        )

    # Quantize the nominal shot and its dissolve handle independently.  The
    # handle therefore seeks to the exact canonical source boundary instead of
    # inheriting up to one frame of cumulative duration-rounding error.
    base = _video_chain(
        source_index=segment.source_index,
        source_start=segment.source_start,
        source_duration=segment.effective_source_duration,
        timeline_duration=segment.duration,
        crop=segment.crop,
        pan_crop=segment.pan_crop,
        editorial_transform=segment.editorial_transform,
        look_adjustment=segment.look_adjustment,
        boundary_look_track=segment.boundary_look_track,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        output_label=f"vb{index}",
    )
    handle_crop = segment.crop
    handle_editorial_transform: EditorialTransformTrack | None = None
    if segment.pan_crop is not None:
        track = segment.pan_crop.simplified()
        crop_width, crop_height = track.fixed_crop_size
        _time, left, top = track.fixed_crop_positions()[-1]
        handle_crop = NormalizedCrop(left, top, crop_width, crop_height)
    if segment.editorial_transform is not None:
        final = segment.editorial_transform.final_keyframe
        handle_editorial_transform = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(
                    0.0,
                    final.zoom,
                    final.center_x,
                    final.center_y,
                    final.roll_degrees,
                ),
            ),
            reason=f"{segment.editorial_transform.reason}_dissolve_handle_hold",
        )
    handle = _video_chain(
        source_index=segment.source_index,
        source_start=segment.source_start + segment.effective_source_duration,
        source_duration=(extension_frames / fps) * segment.source_playback_rate,
        timeline_duration=extension_frames / fps,
        crop=handle_crop,
        pan_crop=None,
        editorial_transform=handle_editorial_transform,
        look_adjustment=segment.look_adjustment,
        # A dissolve handle is outside the nominal hard-cut segment.  Reusing
        # a local time expression here would restart an incoming ramp at t=0;
        # hold the source-global look only instead.
        boundary_look_track=None,
        width=width,
        height=height,
        fps=fps,
        frame_count=extension_frames,
        output_label=f"vh{index}",
    )
    return (
        f"{base};\n{handle};\n"
        f"[vb{index}][vh{index}]concat=n=2:v=1:a=0[v{index}]"
    )


@dataclass(frozen=True, slots=True)
class _VideoRenderUnit:
    """One independently timed FFmpeg chain.

    ``RenderPlan`` keeps every authored boundary for QA and provenance.  The
    renderer may nevertheless decode a pixel-identical, contiguous same-source
    run as one unit.  This prevents each local ``trim -> setpts -> fps`` chain
    from independently rounding the shared endpoint and emitting the preceding
    source frame twice.
    """

    first_segment_index: int
    last_segment_index: int
    segment: RenderSegment
    frame_count: int


def _coalescing_look(
    adjustment: SourceLookAdjustment | None,
) -> SourceLookAdjustment | None:
    if adjustment is None or adjustment.is_identity:
        return None
    return adjustment


def _has_identity_geometry(segment: RenderSegment) -> bool:
    return (
        segment.crop is None
        and segment.pan_crop is None
        and segment.editorial_transform is None
    )


def _boundary_look_values(
    track: BoundaryLookTrack | None,
    time_seconds: float,
) -> tuple[float, float, float, float]:
    if track is None:
        return (0.0, 1.0, 1.0, 0.0)
    value = track.evaluate(time_seconds)
    return (
        value.brightness,
        value.contrast,
        value.saturation,
        value.temperature_balance,
    )


def _boundary_look_is_frame_continuous(
    left: RenderSegment,
    right: RenderSegment,
    *,
    left_frame_count: int,
    fps: float,
) -> bool:
    left_last_time = min(left.duration, (left_frame_count - 1) / fps)
    outgoing = _boundary_look_values(left.boundary_look_track, left_last_time)
    incoming = _boundary_look_values(right.boundary_look_track, 0.0)
    return all(
        math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=1e-9)
        for left_value, right_value in zip(outgoing, incoming, strict=True)
    )


def _can_coalesce_identity_run(
    plan: RenderPlan,
    frame_counts: tuple[int, ...],
    *,
    first_index: int,
    candidate_last_index: int,
) -> bool:
    """Return whether one same-source run is provably pixel-equivalent.

    Only hard-cut boundaries with a continuous source clock and no per-segment
    geometry are eligible.  Static source look must be identical, and local
    boundary-look tracks must meet at the same frame-evaluated state so their
    keyframes can be shifted onto the shared clock.  A very small internal
    linear-clock error allowance covers measured piecewise time-warp noise
    while remaining far below a source or output frame.
    """

    if candidate_last_index <= first_index:
        return True
    segments = plan.segments[first_index : candidate_last_index + 1]
    first = segments[0]
    if not _has_identity_geometry(first):
        return False
    first_look = _coalescing_look(first.look_adjustment)
    for offset, segment in enumerate(segments):
        if (
            segment.source_index != first.source_index
            or not _has_identity_geometry(segment)
            or _coalescing_look(segment.look_adjustment) != first_look
        ):
            return False
        if offset == 0:
            continue
        boundary_index = first_index + offset - 1
        transition = plan.transitions[boundary_index]
        previous = segments[offset - 1]
        if (
            transition.kind is not TransitionKind.HARD_CUT
            or abs(previous.source_end - segment.source_start)
            > MAXIMUM_COALESCED_SOURCE_GAP_SECONDS
            or not _boundary_look_is_frame_continuous(
                previous,
                segment,
                left_frame_count=frame_counts[boundary_index],
                fps=plan.fps,
            )
        ):
            return False

    timeline_duration = sum(segment.duration for segment in segments)
    source_duration = segments[-1].source_end - first.source_start
    cumulative_timeline = 0.0
    maximum_clock_error = (
        MAXIMUM_COALESCED_INTERNAL_CLOCK_ERROR_FRAMES / plan.fps
    )
    for segment in segments[:-1]:
        cumulative_timeline += segment.duration
        linear_source_boundary = (
            first.source_start
            + source_duration * cumulative_timeline / timeline_duration
        )
        if abs(linear_source_boundary - segment.source_end) > maximum_clock_error:
            return False
    return True


def _combined_boundary_look_track(
    segments: tuple[RenderSegment, ...],
) -> BoundaryLookTrack | None:
    """Shift compatible segment-local look tracks onto one run clock."""

    identity = BoundaryLookKeyframe(0.0)
    keyframes: list[BoundaryLookKeyframe] = []
    timeline_offset = 0.0
    for segment in segments:
        track = segment.boundary_look_track
        if track is None or track.is_identity:
            if not keyframes:
                keyframes.append(identity)
        else:
            for keyframe in track.keyframes:
                shifted = BoundaryLookKeyframe(
                    time_seconds=timeline_offset + keyframe.time_seconds,
                    brightness=keyframe.brightness,
                    contrast=keyframe.contrast,
                    saturation=keyframe.saturation,
                    temperature_balance=keyframe.temperature_balance,
                )
                if (
                    keyframes
                    and shifted.time_seconds
                    <= keyframes[-1].time_seconds + 1e-12
                ):
                    # At an exact segment boundary the incoming frame owns the
                    # timestamp.  Eligibility already proved both endpoint
                    # states equal, so replacing the prior sample is neutral.
                    keyframes[-1] = shifted
                else:
                    keyframes.append(shifted)
        timeline_offset += segment.duration
    if not keyframes or keyframes[0].time_seconds > 1e-12:
        keyframes.insert(0, identity)
    combined = BoundaryLookTrack(tuple(keyframes))
    return None if combined.is_identity else combined


def _video_render_units(
    plan: RenderPlan,
    frame_counts: tuple[int, ...],
) -> tuple[_VideoRenderUnit, ...]:
    """Coalesce only safely equivalent identity same-source hard-cut runs."""

    units: list[_VideoRenderUnit] = []
    first_index = 0
    while first_index < len(plan.segments):
        last_index = first_index
        while (
            last_index + 1 < len(plan.segments)
            and _can_coalesce_identity_run(
                plan,
                frame_counts,
                first_index=first_index,
                candidate_last_index=last_index + 1,
            )
        ):
            last_index += 1
        first = plan.segments[first_index]
        last = plan.segments[last_index]
        if last_index == first_index:
            combined = first
        else:
            combined_segments = plan.segments[first_index : last_index + 1]
            combined = RenderSegment(
                source_index=first.source_index,
                source_start=first.source_start,
                duration=sum(
                    segment.duration
                    for segment in combined_segments
                ),
                master_start=first.master_start,
                source_duration=last.source_end - first.source_start,
                look_adjustment=_coalescing_look(first.look_adjustment),
                boundary_look_track=_combined_boundary_look_track(
                    combined_segments
                ),
            )
        units.append(
            _VideoRenderUnit(
                first_segment_index=first_index,
                last_segment_index=last_index,
                segment=combined,
                frame_count=sum(frame_counts[first_index : last_index + 1]),
            )
        )
        first_index = last_index + 1
    return tuple(units)


def build_filter_graph(plan: RenderPlan) -> str:
    frame_counts = segment_frame_counts(plan)
    units = _video_render_units(plan, frame_counts)
    parts = [
        _video_filter(
            unit.first_segment_index,
            unit.segment,
            plan.width,
            plan.height,
            plan.fps,
            unit.frame_count,
            (
                plan.transitions[unit.last_segment_index].frame_count
                if unit.last_segment_index < len(plan.transitions)
                and plan.transitions[unit.last_segment_index].kind
                is TransitionKind.FACE_DISSOLVE
                else 0
            ),
        )
        for unit in units
    ]
    # Build each maximal run of dissolve-linked shots as one xfade chain, then
    # concatenate those runs across ordinary hard cuts.  Extending only the
    # outgoing side by the overlap makes every xfade duration-neutral.
    chunk_labels: list[str] = []
    unit_index = 0
    xfade_index = 0
    while unit_index < len(units):
        unit = units[unit_index]
        current_label = f"v{unit.first_segment_index}"
        cumulative_frames = unit.frame_count
        while (
            unit.last_segment_index < len(plan.transitions)
            and plan.transitions[unit.last_segment_index].kind
            is TransitionKind.FACE_DISSOLVE
        ):
            transition = plan.transitions[unit.last_segment_index]
            next_unit_index = unit_index + 1
            next_unit = units[next_unit_index]
            output_label = f"vx{xfade_index}"
            parts.append(
                f"[{current_label}][v{next_unit.first_segment_index}]"
                "xfade=transition=custom:"
                f"expr='{_FACE_DISSOLVE_XFADE_EXPRESSION}':"
                f"duration={transition.duration_seconds(plan.fps):.9f}:"
                f"offset={cumulative_frames / plan.fps:.9f},format=yuv420p"
                f"[{output_label}]"
            )
            current_label = output_label
            cumulative_frames += next_unit.frame_count
            xfade_index += 1
            unit_index = next_unit_index
            unit = next_unit
        chunk_labels.append(current_label)
        unit_index += 1
    if len(chunk_labels) == 1:
        parts.append(f"[{chunk_labels[0]}]null[vcat]")
    else:
        labels = "".join(f"[{label}]" for label in chunk_labels)
        parts.append(f"{labels}concat=n={len(chunk_labels)}:v=1:a=0[vcat]")
    parts.append(
        f"[vcat]fps={plan.fps:.6f},tpad=stop_mode=clone:stop_duration=1,"
        f"trim=end_frame={plan.output_frame_count},"
        f"setpts=N/({plan.fps:.6f}*TB),settb=AVTB[vout]"
    )
    parts.append(
        f"[{plan.reference_source}:a]atrim=start={plan.master_audio_start:.6f}:"
        f"duration={plan.render_duration:.6f},"
        "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0[aout]"
    )
    return ";\n".join(parts)


def render(
    plan: RenderPlan,
    *,
    ffmpeg: str = "ffmpeg",
    video_codec: str = "h264_videotoolbox",
) -> str:
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary output on the destination volume so the final replace
    # is atomic even when the user renders to an external SSD.
    with tempfile.TemporaryDirectory(
        prefix=".ive-crosscut-", dir=plan.output.parent
    ) as temp:
        root = Path(temp)
        filter_path = root / "filter.ffscript"
        filter_path.write_text(build_filter_graph(plan), encoding="utf-8")
        temporary_output = root / f"render{plan.output.suffix or '.mp4'}"
        codecs = (video_codec, "libx264") if video_codec == "h264_videotoolbox" else (video_codec,)
        failures: list[str] = []
        for codec in codecs:
            command: list[str] = [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
            ]
            for source in plan.sources:
                command.extend(("-i", str(source)))
            command.extend(
                (
                    "-filter_complex_script",
                    str(filter_path),
                    "-map",
                    "[vout]",
                    "-map",
                    "[aout]",
                    "-c:v",
                    codec,
                )
            )
            if codec == "libx264":
                # A quality-targeted software fallback avoids wasting bits on
                # static shots while retaining fine stage lighting, confetti,
                # and hair detail in high-motion sections.
                command.extend(("-preset", "slow", "-crf", "16", "-profile:v", "high"))
            else:
                # VideoToolbox has no CRF-equivalent quality mode. Give the
                # hardware encoder enough headroom for dense choreography and
                # broadcast lighting while retaining a compatible H.264 file.
                command.extend(
                    (
                        "-b:v",
                        "24M",
                        "-maxrate",
                        "36M",
                        "-bufsize",
                        "48M",
                        "-profile:v",
                        "high",
                    )
                )
            command.extend(
                (
                    "-tag:v",
                    "avc1",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "320k",
                    "-movflags",
                    "+faststart",
                    str(temporary_output),
                )
            )
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                temporary_output.replace(plan.output)
                return codec
            failures.append(f"{codec}: {completed.stderr.strip() or 'unknown FFmpeg error'}")
        raise RuntimeError("video render failed: " + " | ".join(failures))


def write_plan(plan: RenderPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")


def contiguous_segments(segments: Iterable[RenderSegment]) -> tuple[RenderSegment, ...]:
    return tuple(sorted(segments, key=lambda segment: segment.master_start))
