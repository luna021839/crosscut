"""FFmpeg render-plan construction for synchronized cross-edits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Callable, Iterable

from .ive import AspectRatio
from .timewarp import TimeWarpMap
from .render_frame_schedule import (
    FRAME_SCHEDULE_CONTRACT, VerifiedRenderFrameClock,
    probe_render_frame_clock, sampler_frame_schedule,
)


RENDER_PLAN_SOURCE_INDEX_DOMAIN = "render_plan_sources_array_index"
MAXIMUM_RENDER_TIMEWARP_RATE_DELTA = 0.02
# Joining a contiguous same-source run may replace several locally linear
# time-warps with one linear decode clock.  Keep every serialized internal
# endpoint within 1/500th of one output frame so the optimization cannot hide
# a real source-clock edit or meaningfully move a broadcaster-native camera
# cut.
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
# This is the renderer's physical envelope, not the default editorial style.
# Individual planners retain their narrower motion limits.  A decoded-output
# inverse correction may use the wider envelope only when source resolution
# proves the crop is lossless at delivery size.
EDITORIAL_TRANSFORM_MAXIMUM_ZOOM = 2.0
EDITORIAL_TRANSFORM_MAXIMUM_CENTER_SHIFT = 0.25


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
        if not 1.0 <= self.zoom <= EDITORIAL_TRANSFORM_MAXIMUM_ZOOM:
            raise ValueError(
                "editorial transform zoom must be in "
                f"[1.0, {EDITORIAL_TRANSFORM_MAXIMUM_ZOOM:.2f}]"
            )
        if not 0.0 <= self.center_x <= 1.0 or not 0.0 <= self.center_y <= 1.0:
            raise ValueError("editorial transform center must be normalized")
        if (
            math.hypot(self.center_x - 0.5, self.center_y - 0.5)
            > EDITORIAL_TRANSFORM_MAXIMUM_CENTER_SHIFT + 1e-9
        ):
            raise ValueError("editorial transform pan exceeds the quarter-frame cap")
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


def _overlap_transform_is_frame_neutral(
    left: EditorialTransformTrack | None,
    right: EditorialTransformTrack | None,
    *,
    overlap_frame_count: int,
    fps: float,
) -> bool:
    """Prove that an overlap never sees a virtual-camera transform.

    The outgoing picture is an extension handle held at the authored final
    state.  The incoming picture starts at local time zero.  A transform
    elsewhere in either segment is therefore safe when the outgoing handle
    and every incoming overlap frame remain exactly identity.  Sampling the
    inclusive end timestamp also prevents a transform from starting on the
    effect's terminal frame because of FFmpeg time-base rounding.
    """

    identity = (1.0, 0.5, 0.5, 0.0)

    def is_identity(values: tuple[float, float, float, float]) -> bool:
        return all(
            math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-9)
            for value, expected in zip(values, identity, strict=True)
        )

    if left is not None and not is_identity(
        (
            left.final_keyframe.zoom,
            left.final_keyframe.center_x,
            left.final_keyframe.center_y,
            left.final_keyframe.roll_degrees,
        )
    ):
        return False
    return all(
        is_identity(_editorial_state_values(right, frame_index / fps))
        for frame_index in range(overlap_frame_count + 1)
    )


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
    """Static, duration-neutral source look correction consumed by FFmpeg.

    The ordinary synchronized-source matcher still uses its tighter production
    defaults.  The wider serialization envelope also accommodates automatic
    member cache calibration. Its bounds cover the equivalent composition of
    the empirically validated three-pass member grade while the renderer emits
    one duration-neutral pass. They remain far inside FFmpeg's legal ranges and
    never alter timing.
    """

    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    temperature_balance: float = 0.0
    tint_balance: float = 0.0
    detail_enhancement: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.brightness,
            self.contrast,
            self.saturation,
            self.temperature_balance,
            self.tint_balance,
            self.detail_enhancement,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("source look adjustment values must be finite")
        if not -0.18 <= self.brightness <= 0.18:
            raise ValueError("source look brightness exceeds the calibrated bound")
        if not 0.60 <= self.contrast <= 1.40:
            raise ValueError("source look contrast exceeds the calibrated bound")
        if not 0.80 <= self.saturation <= 1.20:
            raise ValueError("source look saturation exceeds the calibrated bound")
        if not -0.18 <= self.temperature_balance <= 0.18:
            raise ValueError(
                "source look temperature balance exceeds the calibrated bound"
            )
        if not -0.10 <= self.tint_balance <= 0.10:
            raise ValueError("source look tint balance exceeds the calibrated bound")
        if not 0.0 <= self.detail_enhancement <= 1.0:
            raise ValueError(
                "source look detail enhancement exceeds the calibrated bound"
            )

    @property
    def is_identity(self) -> bool:
        return (
            abs(self.brightness) <= 1e-12
            and abs(self.contrast - 1.0) <= 1e-12
            and abs(self.saturation - 1.0) <= 1e-12
            and abs(self.temperature_balance) <= 1e-12
            and abs(self.tint_balance) <= 1e-12
            and abs(self.detail_enhancement) <= 1e-12
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
        values = (
            self.brightness,
            self.contrast,
            self.saturation,
            self.temperature_balance,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("boundary look values must be finite")
        if not -0.05 <= self.brightness <= 0.05:
            raise ValueError("boundary look brightness exceeds its local bound")
        if not 0.94 <= self.contrast <= 1.06:
            raise ValueError("boundary look contrast exceeds its local bound")
        if not 0.94 <= self.saturation <= 1.06:
            raise ValueError("boundary look saturation exceeds its local bound")
        if not -0.03 <= self.temperature_balance <= 0.03:
            raise ValueError("boundary look temperature exceeds its local bound")

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
    # Separately timed outgoing overlap camera. None preserves legacy final
    # keyframe hold; explicit tracks start at the source-local segment end.
    outgoing_overlap_transform: EditorialTransformTrack | None = None

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
        if self.outgoing_overlap_transform is not None and not isinstance(
            self.outgoing_overlap_transform, EditorialTransformTrack
        ):
            raise TypeError("outgoing overlap camera must be an editorial track")

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


def _tier_d_base_crop_discontinuity(
    left: RenderSegment,
    right: RenderSegment,
    *,
    left_frame_count: int,
    right_frame_count: int,
    fps: float,
) -> str | None:
    frame = 1.0 / fps
    left_last_time = min(left.duration, (left_frame_count - 1) * frame)
    outgoing = _base_crop_values(left, left_last_time)
    authored_endpoint = _base_crop_values(left, left.duration)
    incoming = _base_crop_values(right, 0.0)
    # Tier D is an invisible continuation of one source-local shot.  Its
    # authored crop must therefore meet exactly at the mathematical boundary;
    # this distinguishes a real reset from the harmless variable-size final
    # frame interval introduced by global frame-count quantization.
    endpoint_deltas = tuple(
        abs(left_value - right_value)
        for left_value, right_value in zip(
            authored_endpoint,
            incoming,
            strict=True,
        )
    )
    if max(endpoint_deltas) > 1e-8:
        return (
            "authored_endpoint_delta="
            f"{max(endpoint_deltas):.9f}"
        )
    maximum_steps = (
        0.04 / (0.55 * fps),
        0.04 / (0.55 * fps),
        0.18 / (0.55 * fps),
        0.18 / (0.55 * fps),
    )
    coordinate_names = ("x", "y", "width", "height")
    for index, (coordinate_name, maximum_step) in enumerate(
        zip(coordinate_names, maximum_steps, strict=True)
    ):
        cross_step = incoming[index] - outgoing[index]
        if abs(cross_step) > maximum_step + 1e-8:
            return (
                f"frame_step_{coordinate_name}={abs(cross_step):.9f};"
                f"maximum={maximum_step:.9f}"
            )
    return None


def _tier_d_base_crop_is_frame_continuous(
    left: RenderSegment,
    right: RenderSegment,
    *,
    left_frame_count: int,
    right_frame_count: int,
    fps: float,
) -> bool:
    return _tier_d_base_crop_discontinuity(
        left,
        right,
        left_frame_count=left_frame_count,
        right_frame_count=right_frame_count,
        fps=fps,
    ) is None


class TransitionKind(str, Enum):
    """Video transition applied at one nominal segment boundary."""

    HARD_CUT = "hard_cut"
    FACE_DISSOLVE = "face_dissolve"
    MEMBER_ZOOM_BRIDGE = "member_zoom_bridge"
    MEMBER_BLUR_BRIDGE = "member_blur_bridge"

    @property
    def uses_overlap(self) -> bool:
        return self is not TransitionKind.HARD_CUT


MEMBER_ZOOM_BRIDGE_REASON = (
    "official_target_member_required_zoom_in_bridge_v1"
)
MEMBER_BLUR_BRIDGE_REASON = (
    "official_target_member_required_horizontal_blur_bridge_v1"
)
DIRECT_MEMBER_POSE_DISSOLVE_REASON = (
    "official_target_member_pose_matched_logistic_face_dissolve_v1"
)
MEMBER_SOURCE_MEASURED_JOINT_OVERLAP_AUTHORITY = (
    "source_measured_joint_registration_pending_pixel_review_v1"
)
MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY = (
    "decoded_hardcut_output_endpoint_match_effect_proxy_revalidation_required_v1"
)


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

    Every overlap transition uses an outgoing-only source handle. The incoming
    shot still begins at the nominal cut, so both pictures show the same
    canonical song time and the edit's total frame count is unchanged.
    """

    kind: TransitionKind = TransitionKind.HARD_CUT
    frame_count: int = 0
    reason: str = "default_hard_cut"
    face_mismatch: float | None = None
    tier: str | None = None
    appearance_min_cosine_similarity: float | None = None
    future_pose_max_mismatch: float | None = None
    future_silhouette_max_mismatch: float | None = None
    transform_overlap_authority: str | None = None

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
        if self.kind.uses_overlap and self.frame_count < 3:
            raise ValueError("overlap transitions need at least three frames")
        if not self.reason:
            raise ValueError("transition reason cannot be blank")
        if self.transform_overlap_authority is not None and not (
            self.kind is TransitionKind.FACE_DISSOLVE
            and self.reason == DIRECT_MEMBER_POSE_DISSOLVE_REASON
            and self.transform_overlap_authority in {
                MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY,
                MEMBER_SOURCE_MEASURED_JOINT_OVERLAP_AUTHORITY,
            }
        ):
            raise ValueError(
                "transform overlap authority is valid only for an explicitly "
                "reviewed or source-measured direct-member dissolve"
            )
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
        payload: dict[str, object] = {
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
                else "ffmpeg_xfade_zoomin_v1"
                if self.kind is TransitionKind.MEMBER_ZOOM_BRIDGE
                else "ffmpeg_xfade_hblur_v1"
                if self.kind is TransitionKind.MEMBER_BLUR_BRIDGE
                else None
            ),
        }
        if self.transform_overlap_authority is not None:
            payload["transform_overlap_authority"] = (
                self.transform_overlap_authority
            )
        return payload


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
    member_pose_dissolve_requested: bool | None = None
    member_pose_dissolve_effective: bool | None = None
    member_pose_dissolve_count: int | None = None

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
        member_bridge_reasons = {
            TransitionKind.MEMBER_ZOOM_BRIDGE: MEMBER_ZOOM_BRIDGE_REASON,
            TransitionKind.MEMBER_BLUR_BRIDGE: MEMBER_BLUR_BRIDGE_REASON,
        }
        actual_member_bridges = sum(
            transition.kind in member_bridge_reasons
            for transition in self.transitions
        )
        if any(
            transition.kind in member_bridge_reasons
            and transition.reason != member_bridge_reasons[transition.kind]
            for transition in self.transitions
        ):
            raise ValueError(
                "direct member render contains an uncontracted cinematic bridge"
            )
        member_dissolve_fields = (
            self.member_pose_dissolve_requested,
            self.member_pose_dissolve_effective,
            self.member_pose_dissolve_count,
        )
        if any(value is not None for value in member_dissolve_fields):
            if not all(value is not None for value in member_dissolve_fields):
                raise ValueError(
                    "member pose dissolve render metadata must be complete"
                )
            if not isinstance(
                self.member_pose_dissolve_requested,
                bool,
            ) or not isinstance(self.member_pose_dissolve_effective, bool):
                raise TypeError("member pose dissolve render flags must be boolean")
            if (
                isinstance(self.member_pose_dissolve_count, bool)
                or not isinstance(self.member_pose_dissolve_count, int)
                or self.member_pose_dissolve_count < 0
            ):
                raise ValueError("member pose dissolve render count is invalid")
            actual_member_dissolves = sum(
                transition.kind is TransitionKind.FACE_DISSOLVE
                and transition.reason
                == DIRECT_MEMBER_POSE_DISSOLVE_REASON
                for transition in self.transitions
            )
            actual_face_dissolves = sum(
                transition.kind is TransitionKind.FACE_DISSOLVE
                for transition in self.transitions
            )
            if actual_face_dissolves != actual_member_dissolves:
                raise ValueError(
                    "direct member render contains an uncontracted face dissolve"
                )
            if self.member_pose_dissolve_count != actual_member_dissolves:
                raise ValueError(
                    "member pose dissolve render count disagrees with transitions"
                )
            if self.member_pose_dissolve_effective != bool(actual_member_dissolves):
                raise ValueError(
                    "member pose dissolve effective flag disagrees with transitions"
                )
            if not self.member_pose_dissolve_requested and (
                self.member_pose_dissolve_effective
                or self.member_pose_dissolve_count != 0
                or actual_face_dissolves != 0
                or actual_member_bridges != 0
            ):
                raise ValueError(
                    "disabled member pose dissolve cannot author an effect"
                )
        elif actual_member_bridges:
            raise ValueError(
                "member cinematic bridge requires complete request metadata"
            )
        counts = segment_frame_counts(self)
        for index, segment in enumerate(self.segments):
            handle = segment.outgoing_overlap_transform
            if handle is None:
                continue
            if index >= len(self.transitions) or not self.transitions[index].kind.uses_overlap:
                raise ValueError("outgoing camera handle needs an overlap transition")
            transition = self.transitions[index]
            if transition.transform_overlap_authority not in {
                MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY,
                MEMBER_SOURCE_MEASURED_JOINT_OVERLAP_AUTHORITY,
            }:
                raise ValueError("outgoing camera handle needs explicit registration authority")
            if (handle.keyframes[0].time_seconds != 0.0 or
                handle.final_keyframe.time_seconds > transition.frame_count/self.fps+1e-7 or
                handle.final_keyframe.time_seconds < (transition.frame_count-1)/self.fps-1e-7):
                raise ValueError("outgoing camera handle must cover exactly its output frames")
            if segment.editorial_transform is None:
                raise ValueError("outgoing camera handle needs a continuous nominal camera")
            before=segment.editorial_transform.final_keyframe
            after=handle.keyframes[0]
            if any(abs(getattr(before,name)-getattr(after,name))>1e-7
                   for name in ('zoom','center_x','center_y','roll_degrees')):
                raise ValueError("outgoing camera handle cannot reset the nominal camera")
            for frame in range(transition.frame_count):
                handle.evaluate(frame/self.fps)
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
            if transition.transform_overlap_authority == MEMBER_SOURCE_MEASURED_JOINT_OVERLAP_AUTHORITY and (
                self.segments[index].outgoing_overlap_transform is None
                or self.segments[index+1].editorial_transform is None
            ):
                raise ValueError("joint registration needs both explicit overlap cameras")
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
            crop_discontinuity = (
                _tier_d_base_crop_discontinuity(
                    self.segments[index],
                    self.segments[index + 1],
                    left_frame_count=counts[index],
                    right_frame_count=counts[index + 1],
                    fps=self.fps,
                )
                if transition.tier == "D"
                else None
            )
            if crop_discontinuity is not None:
                raise ValueError(
                    "same-source hold cannot reset its base crop: "
                    f"boundary={index}; {crop_discontinuity}"
                )
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
                transition.kind.uses_overlap
                and not _overlap_transform_is_frame_neutral(
                    self.segments[index].editorial_transform,
                    self.segments[index + 1].editorial_transform,
                    overlap_frame_count=transition.frame_count,
                    fps=self.fps,
                )
                and transition.transform_overlap_authority not in {
                    MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY,
                    MEMBER_SOURCE_MEASURED_JOINT_OVERLAP_AUTHORITY,
                }
            ):
                raise ValueError(
                    "overlap transition cannot use an unreviewed editorial transform"
                )
            if transition.kind.uses_overlap and not (
                left_endpoint_identity and right_endpoint_identity
            ):
                raise ValueError(
                    "overlap transition cannot use a hard-cut boundary look ramp"
                )
            if (
                transition.kind.uses_overlap
                and transition.frame_count > counts[index + 1]
            ):
                raise ValueError(
                    "overlap transition cannot exceed the incoming segment"
                )
            if (
                transition.kind.uses_overlap
                and self.segments[index].source_index
                == self.segments[index + 1].source_index
            ):
                raise ValueError("overlap transition requires a real source change")

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
            if segment.outgoing_overlap_transform is None:
                payload.pop("outgoing_overlap_transform", None)
            serialized_look = payload.get("look_adjustment")
            if (
                isinstance(serialized_look, dict)
                and abs(float(serialized_look.get("tint_balance", 0.0))) <= 1e-12
            ):
                # Preserve exact replay/hash compatibility with v1-v2 plans.
                serialized_look.pop("tint_balance", None)
            if (
                isinstance(serialized_look, dict)
                and abs(
                    float(serialized_look.get("detail_enhancement", 0.0))
                )
                <= 1e-12
            ):
                # Preserve exact replay/hash compatibility with plans written
                # before decoded-output detail equalization existed.
                serialized_look.pop("detail_enhancement", None)
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
            "member_pose_dissolve_requested": (
                self.member_pose_dissolve_requested
            ),
            "member_pose_dissolve_effective": (
                self.member_pose_dissolve_effective
            ),
            "member_pose_dissolve_count": self.member_pose_dissolve_count,
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


def quantized_segment_frame_counts(
    segments: Sequence[RenderSegment], fps: float
) -> tuple[int, ...]:
    """Shared output clock for rendering and frame-visible motion evidence."""
    if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("output fps must be finite and positive")
    counts: list[int] = []
    cumulative_time = 0.0
    previous_boundary = 0
    for segment in segments:
        cumulative_time += segment.duration
        boundary = int(round(cumulative_time * fps))
        count = boundary - previous_boundary
        if count < 2:
            raise ValueError(
                f"segment at {segment.master_start:.6f}s is shorter than two output frames"
            )
        counts.append(count)
        previous_boundary = boundary
    return tuple(counts)


def segment_frame_counts(plan: RenderPlan) -> tuple[int, ...]:
    """Quantize global cut boundaries once so per-cut rounding cannot drift."""

    counts = quantized_segment_frame_counts(plan.segments, plan.fps)
    if sum(counts) != plan.output_frame_count:
        raise ValueError("quantized segment frames do not match the output duration")
    return counts


def _editorial_transform_filters(
    track: EditorialTransformTrack,
    *,
    width: int,
    height: int,
    fps: float,
    input_clock_is_canonical: bool = False,
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
    maximum_zoom = max(keyframe.zoom for keyframe in track.keyframes)
    # Preserve source detail through the authored camera move. Scaling to the
    # final canvas before zoompan would discard 4K pixels and then enlarge the
    # already-downsampled 1080p frame. The larger even working canvas makes a
    # zoom of Z consume width*Z by height*Z source pixels before the final
    # output sample. Direct-member selection independently proves that this
    # canvas fits inside its post-aspect-fill source geometry.
    working_width = max(
        width,
        int(math.ceil(width * maximum_zoom / 2.0)) * 2,
    )
    working_height = max(
        height,
        int(math.ceil(height * maximum_zoom / 2.0)) * 2,
    )
    filters = [
        (
            f"scale={working_width}:{working_height}:"
            "force_original_aspect_ratio=increase"
        ),
        f"crop={working_width}:{working_height}",
        "setsar=1",
    ]
    if not input_clock_is_canonical:
        # Canonicalize after local time-warp before zoompan. zoompan emits one
        # output frame per input frame, so the small source-rate correction is
        # retained instead of being silently replaced by source cadence.
        filters.append(f"fps={fps:.6f}")
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
    if (
        abs(adjustment.temperature_balance) > 1e-12
        or abs(adjustment.tint_balance) > 1e-12
    ):
        warm = adjustment.temperature_balance
        tint = adjustment.tint_balance
        # Keep source look calibration in native YUV and copy luma exactly.
        # FFmpeg colorbalance's preserve-lightness mode is not numerically
        # safe for tiny all-range RGB offsets on saturated LED backgrounds: it
        # can turn smooth blue/magenta walls into large gray posterized blocks
        # even though the stream still decodes without errors.  These two
        # affine chroma axes approximate the former RGB intent while remaining
        # sample-exact on Y and continuous for every legal adjustment.
        cb_offset = -86.0 * warm - 64.0 * tint
        cr_offset = 74.0 * warm - 80.0 * tint
        filters.append(
            "geq="
            "lum='lum(X,Y)':"
            f"cb='clip(cb(X,Y){cb_offset:+.6f},0,255)':"
            f"cr='clip(cr(X,Y){cr_offset:+.6f},0,255)'"
        )
    if adjustment.detail_enhancement > 1e-12:
        # Contrast Adaptive Sharpen raises local detail without the broad halos
        # of a fixed unsharp mask. It is enabled only by decoded-output member
        # calibration and the resulting pixels still have to pass the same
        # perceptual proxy and final source-retention gates.
        filters.append(f"cas=strength={adjustment.detail_enhancement:.6f}")
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


def _video_post_clock_filters(
    *,
    crop: NormalizedCrop | None,
    pan_crop: PanCropTrack | None,
    editorial_transform: EditorialTransformTrack | None,
    look_adjustment: SourceLookAdjustment | None,
    boundary_look_track: BoundaryLookTrack | None,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    input_clock_is_canonical: bool,
) -> list[str]:
    """Apply one segment's authored pixels on its local output clock."""

    chain: list[str] = []
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
                input_clock_is_canonical=input_clock_is_canonical,
            )
        )
    else:
        chain.extend(
            [
                f"scale={width}:{height}:force_original_aspect_ratio=decrease",
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
                "setsar=1",
            ]
        )
        if not input_clock_is_canonical:
            chain.append(f"fps={fps:.6f}")
    # Source and boundary look operations are spatially uniform and do not
    # alter geometry or cadence.  Apply them after crop/scale/zoompan so a
    # 480px review does not run affine chroma ``geq`` over every 4K source
    # pixel.  This also keeps proxy and delivery rendering on the same exact
    # output-domain color path while reducing work by up to an order of
    # magnitude.  The boundary residual remains after the source-global grade.
    chain.extend(_source_look_filters(look_adjustment))
    chain.extend(_boundary_look_filters(boundary_look_track))
    chain.extend(
        [
            "tpad=stop_mode=clone:stop_duration=1",
            f"trim=end_frame={frame_count}",
            f"setpts=N/({fps:.6f}*TB)",
            "settb=AVTB",
            "format=yuv420p",
        ]
    )
    return chain


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
    source_clock: VerifiedRenderFrameClock | None = None,
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
    if source_clock is not None:
        schedule = sampler_frame_schedule(
            source_start=source_start,
            source_playback_rate=source_duration / timeline_duration,
            native_fps=float(source_clock.frame_rate), output_fps=fps,
            frame_count=frame_count, source_frame_count=source_clock.frame_count,
            maximum_terminal_hold_frames=math.ceil(fps),
        )
        chain = list(schedule.ffmpeg_filters(fps))
        chain[0] = f"[{source_index}:v]" + chain[0]
    chain.extend(
        _video_post_clock_filters(
            crop=crop,
            pan_crop=pan_crop,
            editorial_transform=editorial_transform,
            look_adjustment=look_adjustment,
            boundary_look_track=boundary_look_track,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            input_clock_is_canonical=source_clock is not None,
        )
    )
    return ",".join(chain) + f"[{output_label}]"


def _video_source_clock_chain(
    *,
    source_index: int,
    source_start: float,
    source_duration: float,
    timeline_duration: float,
    fps: float,
    frame_count: int,
    output_label: str,
    source_clock: VerifiedRenderFrameClock | None = None,
) -> str:
    """Decode and quantize one continuous source interval exactly once."""

    if source_clock is not None:
        schedule = sampler_frame_schedule(
            source_start=source_start,
            source_playback_rate=source_duration / timeline_duration,
            native_fps=float(source_clock.frame_rate), output_fps=fps,
            frame_count=frame_count, source_frame_count=source_clock.frame_count,
            maximum_terminal_hold_frames=math.ceil(fps),
        )
        chain = list(schedule.ffmpeg_filters(fps))
        chain[0] = f"[{source_index}:v]" + chain[0]
        return ",".join(chain) + f"[{output_label}]"
    playback_pts_scale = timeline_duration / source_duration
    setpts = (
        "setpts=PTS-STARTPTS"
        if math.isclose(playback_pts_scale, 1.0, rel_tol=0.0, abs_tol=1e-9)
        else f"setpts={playback_pts_scale:.12f}*(PTS-STARTPTS)"
    )
    chain = [
        f"[{source_index}:v]trim=start={source_start:.6f}:duration={source_duration:.6f}",
        setpts,
        f"fps={fps:.6f}",
        "tpad=stop_mode=clone:stop_duration=1",
        f"trim=end_frame={frame_count}",
        f"setpts=N/({fps:.6f}*TB)",
        "settb=AVTB",
    ]
    return ",".join(chain) + f"[{output_label}]"


def _dissolve_handle_video_chain(
    index: int,
    segment: RenderSegment,
    *,
    width: int,
    height: int,
    fps: float,
    extension_frames: int,
    source_clock: VerifiedRenderFrameClock | None = None,
) -> str:
    """Render one separately sought outgoing overlap handle."""

    handle_crop = segment.crop
    handle_editorial_transform: EditorialTransformTrack | None = None
    if segment.pan_crop is not None:
        track = segment.pan_crop.simplified()
        crop_width, crop_height = track.fixed_crop_size
        _time, left, top = track.fixed_crop_positions()[-1]
        handle_crop = NormalizedCrop(left, top, crop_width, crop_height)
    if segment.outgoing_overlap_transform is not None:
        handle_editorial_transform = segment.outgoing_overlap_transform
    elif segment.editorial_transform is not None:
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
    return _video_chain(
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
        source_clock=source_clock,
    )


def _video_filter(
    index: int,
    segment: RenderSegment,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    extension_frames: int = 0,
    source_clock: VerifiedRenderFrameClock | None = None,
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
            source_clock=source_clock,
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
        source_clock=source_clock,
    )
    handle = _dissolve_handle_video_chain(
        index,
        segment,
        width=width,
        height=height,
        fps=fps,
        extension_frames=extension_frames,
        source_clock=source_clock,
    )
    return (
        f"{base};\n{handle};\n"
        f"[vb{index}][vh{index}]concat=n=2:v=1:a=0[v{index}]"
    )


@dataclass(frozen=True, slots=True)
class _VideoPostClockSlice:
    """One render-equivalent slice cut from a shared source clock."""

    segment: RenderSegment
    frame_count: int


@dataclass(frozen=True, slots=True)
class _VideoRenderUnit:
    """One independently decoded FFmpeg source clock.

    ``RenderPlan`` keeps every authored boundary for QA and provenance.  The
    renderer may nevertheless decode a contiguous same-source run as one unit.
    This prevents each local ``trim -> setpts -> fps`` chain from independently
    rounding the shared endpoint and emitting the preceding source frame twice.
    Pixel-identical runs retain the compact combined-segment fast path; other
    runs split this shared clock at exact output-frame boundaries before each
    segment's authored geometry and look are applied.
    """

    first_segment_index: int
    last_segment_index: int
    segments: tuple[RenderSegment, ...]
    segment_frame_counts: tuple[int, ...]
    coalesced_segment: RenderSegment | None
    frame_count: int


def _coalescing_look(
    adjustment: SourceLookAdjustment | None,
) -> SourceLookAdjustment | None:
    if adjustment is None or adjustment.is_identity:
        return None
    return adjustment


def _editorial_transform_track_is_identity(
    track: EditorialTransformTrack | None,
) -> bool:
    """Return whether an authored track renders the identity at every frame.

    Joint decoded candidates deliberately retain a process-bound renderer
    track even when every authorized keyframe is neutral.  Treating the mere
    presence of that object as geometry prevented otherwise continuous
    same-source runs from sharing one decode clock and could duplicate the
    split frame.  Linear interpolation between neutral keyframes is neutral,
    so checking every keyframe is an exact proof for the complete track.
    """

    if track is None:
        return True
    return all(
        math.isclose(keyframe.zoom, 1.0, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(
            keyframe.center_x,
            0.5,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and math.isclose(
            keyframe.center_y,
            0.5,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and math.isclose(
            keyframe.roll_degrees,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        for keyframe in track.keyframes
    )


def _has_identity_geometry(segment: RenderSegment) -> bool:
    return (
        segment.crop is None
        and segment.pan_crop is None
        and _editorial_transform_track_is_identity(
            segment.editorial_transform
        )
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


def _can_share_source_clock(
    plan: RenderPlan,
    *,
    first_index: int,
    candidate_last_index: int,
) -> bool:
    """Return whether a same-source run has one provably linear clock.

    Geometry and look are deliberately irrelevant here because they are
    applied only after the shared canonical clock is split.  Hard cuts, source
    continuity, and one linear playback rate remain mandatory.  A very small
    internal error allowance covers measured time-warp noise while remaining
    far below a source or output frame.
    """

    if candidate_last_index <= first_index:
        return True
    segments = plan.segments[first_index : candidate_last_index + 1]
    first = segments[0]
    for offset, segment in enumerate(segments):
        if segment.source_index != first.source_index:
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


def _can_coalesce_identity_run(
    plan: RenderPlan,
    frame_counts: tuple[int, ...],
    *,
    first_index: int,
    candidate_last_index: int,
) -> bool:
    """Return whether one shared-clock run is also pixel-equivalent."""

    if not _can_share_source_clock(
        plan,
        first_index=first_index,
        candidate_last_index=candidate_last_index,
    ):
        return False
    segments = plan.segments[first_index : candidate_last_index + 1]
    first = segments[0]
    first_look = _coalescing_look(first.look_adjustment)
    for offset, segment in enumerate(segments):
        if (
            not _has_identity_geometry(segment)
            or _coalescing_look(segment.look_adjustment) != first_look
        ):
            return False
        if offset == 0:
            continue
        boundary_index = first_index + offset - 1
        previous = segments[offset - 1]
        if not _boundary_look_is_frame_continuous(
            previous,
            segment,
            left_frame_count=frame_counts[boundary_index],
            fps=plan.fps,
        ):
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


def _combined_identity_segment(
    segments: tuple[RenderSegment, ...],
) -> RenderSegment:
    """Represent one pixel-equivalent identity run without its boundaries."""

    first = segments[0]
    last = segments[-1]
    return RenderSegment(
        source_index=first.source_index,
        source_start=first.source_start,
        duration=sum(segment.duration for segment in segments),
        master_start=first.master_start,
        source_duration=last.source_end - first.source_start,
        look_adjustment=_coalescing_look(first.look_adjustment),
        boundary_look_track=_combined_boundary_look_track(segments),
    )


def _can_coalesce_post_clock_identity_slice(
    segments: tuple[RenderSegment, ...],
    frame_counts: tuple[int, ...],
    *,
    first_offset: int,
    candidate_last_offset: int,
    fps: float,
) -> bool:
    """Prove pixel equivalence after a source clock is already canonical."""

    selected = segments[first_offset : candidate_last_offset + 1]
    first = selected[0]
    first_look = _coalescing_look(first.look_adjustment)
    for offset, segment in enumerate(selected):
        if (
            not _has_identity_geometry(segment)
            or _coalescing_look(segment.look_adjustment) != first_look
        ):
            return False
        if offset == 0:
            continue
        previous_offset = first_offset + offset - 1
        if not _boundary_look_is_frame_continuous(
            selected[offset - 1],
            segment,
            left_frame_count=frame_counts[previous_offset],
            fps=fps,
        ):
            return False
    return True


def _post_clock_slices(
    unit: _VideoRenderUnit,
    *,
    fps: float,
) -> tuple[_VideoPostClockSlice, ...]:
    """Collapse only adjacent pixel-equivalent identity segments."""

    slices: list[_VideoPostClockSlice] = []
    first_offset = 0
    while first_offset < len(unit.segments):
        last_offset = first_offset
        while (
            last_offset + 1 < len(unit.segments)
            and _can_coalesce_post_clock_identity_slice(
                unit.segments,
                unit.segment_frame_counts,
                first_offset=first_offset,
                candidate_last_offset=last_offset + 1,
                fps=fps,
            )
        ):
            last_offset += 1
        selected = unit.segments[first_offset : last_offset + 1]
        segment = (
            selected[0]
            if len(selected) == 1
            else _combined_identity_segment(selected)
        )
        slices.append(
            _VideoPostClockSlice(
                segment=segment,
                frame_count=sum(
                    unit.segment_frame_counts[
                        first_offset : last_offset + 1
                    ]
                ),
            )
        )
        first_offset = last_offset + 1
    return tuple(slices)


def _video_render_units(
    plan: RenderPlan,
    frame_counts: tuple[int, ...],
) -> tuple[_VideoRenderUnit, ...]:
    """Group only continuous, linear same-source hard-cut clocks."""

    units: list[_VideoRenderUnit] = []
    first_index = 0
    while first_index < len(plan.segments):
        last_index = first_index
        while (
            last_index + 1 < len(plan.segments)
            and _can_share_source_clock(
                plan,
                first_index=first_index,
                candidate_last_index=last_index + 1,
            )
        ):
            last_index += 1
        first = plan.segments[first_index]
        unit_segments = plan.segments[first_index : last_index + 1]
        unit_frame_counts = frame_counts[first_index : last_index + 1]
        if last_index == first_index:
            combined = first
        elif _can_coalesce_identity_run(
            plan,
            frame_counts,
            first_index=first_index,
            candidate_last_index=last_index,
        ):
            combined = _combined_identity_segment(unit_segments)
        else:
            combined = None
        units.append(
            _VideoRenderUnit(
                first_segment_index=first_index,
                last_segment_index=last_index,
                segments=unit_segments,
                segment_frame_counts=unit_frame_counts,
                coalesced_segment=combined,
                frame_count=sum(unit_frame_counts),
            )
        )
        first_index = last_index + 1
    return tuple(units)


def _shared_clock_video_filter(
    unit: _VideoRenderUnit,
    *,
    width: int,
    height: int,
    fps: float,
    extension_frames: int,
    source_clock: VerifiedRenderFrameClock | None = None,
) -> str:
    """Split one canonical decode clock, then apply segment-local pixels."""

    first = unit.segments[0]
    last = unit.segments[-1]
    post_clock_slices = _post_clock_slices(unit, fps=fps)
    clock_label = f"vc{unit.first_segment_index}"
    parts = [
        _video_source_clock_chain(
            source_index=first.source_index,
            source_start=first.source_start,
            source_duration=last.source_end - first.source_start,
            timeline_duration=sum(segment.duration for segment in unit.segments),
            fps=fps,
            frame_count=unit.frame_count,
            output_label=clock_label,
            source_clock=source_clock,
        )
    ]
    split_labels = tuple(
        f"vcs{unit.first_segment_index}_{offset}"
        for offset in range(len(post_clock_slices))
    )
    parts.append(
        f"[{clock_label}]split={len(split_labels)}"
        + "".join(f"[{label}]" for label in split_labels)
    )

    pixel_labels: list[str] = []
    start_frame = 0
    for offset, post_clock_slice in enumerate(post_clock_slices):
        segment = post_clock_slice.segment
        end_frame = start_frame + post_clock_slice.frame_count
        pixel_label = f"vsp{unit.first_segment_index}_{offset}"
        chain = [
            f"[{split_labels[offset]}]trim="
            f"start_frame={start_frame}:end_frame={end_frame}",
            "setpts=PTS-STARTPTS",
        ]
        chain.extend(
            _video_post_clock_filters(
                crop=segment.crop,
                pan_crop=segment.pan_crop,
                editorial_transform=segment.editorial_transform,
                look_adjustment=segment.look_adjustment,
                boundary_look_track=segment.boundary_look_track,
                width=width,
                height=height,
                fps=fps,
                frame_count=post_clock_slice.frame_count,
                input_clock_is_canonical=True,
            )
        )
        parts.append(",".join(chain) + f"[{pixel_label}]")
        pixel_labels.append(pixel_label)
        start_frame = end_frame

    nominal_label = (
        f"vb{unit.first_segment_index}"
        if extension_frames
        else f"v{unit.first_segment_index}"
    )
    parts.append(
        "".join(f"[{label}]" for label in pixel_labels)
        + f"concat=n={len(pixel_labels)}:v=1:a=0[{nominal_label}]"
    )
    if extension_frames:
        parts.append(
            _dissolve_handle_video_chain(
                unit.first_segment_index,
                last,
                width=width,
                height=height,
                fps=fps,
                extension_frames=extension_frames,
                source_clock=source_clock,
            )
        )
        parts.append(
            f"[{nominal_label}][vh{unit.first_segment_index}]"
            f"concat=n=2:v=1:a=0[v{unit.first_segment_index}]"
        )
    return ";\n".join(parts)


def _video_render_unit_filter(
    unit: _VideoRenderUnit,
    *,
    width: int,
    height: int,
    fps: float,
    extension_frames: int,
    source_clock: VerifiedRenderFrameClock | None = None,
) -> str:
    if unit.coalesced_segment is not None:
        return _video_filter(
            unit.first_segment_index,
            unit.coalesced_segment,
            width,
            height,
            fps,
            unit.frame_count,
            extension_frames,
            source_clock=source_clock,
        )
    return _shared_clock_video_filter(
        unit,
        width=width,
        height=height,
        fps=fps,
        extension_frames=extension_frames,
        source_clock=source_clock,
    )


def build_filter_graph(
    plan: RenderPlan,
    *, source_clocks: dict[int, VerifiedRenderFrameClock] | None = None,
) -> str:
    frame_counts = segment_frame_counts(plan)
    units = _video_render_units(plan, frame_counts)
    parts = [
        _video_render_unit_filter(
            unit,
            width=plan.width,
            height=plan.height,
            fps=plan.fps,
            source_clock=(source_clocks or {}).get(unit.segments[0].source_index),
            extension_frames=(
                plan.transitions[unit.last_segment_index].frame_count
                if unit.last_segment_index < len(plan.transitions)
                and plan.transitions[unit.last_segment_index].kind.uses_overlap
                else 0
            ),
        )
        for unit in units
    ]
    # Build each maximal run of overlap-linked shots as one xfade chain, then
    # concatenate those runs across ordinary hard cuts.  Extending only the
    # outgoing side by the overlap makes every xfade duration-neutral.
    chunk_labels: list[str] = []
    unit_index = 0
    xfade_index = 0
    while unit_index < len(units):
        unit = units[unit_index]
        current_label = f"v{unit.first_segment_index}"
        cumulative_frames = unit.frame_count
        chunk_had_overlap = False
        while (
            unit.last_segment_index < len(plan.transitions)
            and plan.transitions[unit.last_segment_index].kind.uses_overlap
        ):
            transition = plan.transitions[unit.last_segment_index]
            chunk_had_overlap = True
            next_unit_index = unit_index + 1
            next_unit = units[next_unit_index]
            output_label = f"vx{xfade_index}"
            if transition.kind is TransitionKind.FACE_DISSOLVE:
                xfade = (
                    "xfade=transition=custom:"
                    f"expr='{_FACE_DISSOLVE_XFADE_EXPRESSION}':"
                )
            elif transition.kind is TransitionKind.MEMBER_ZOOM_BRIDGE:
                xfade = "xfade=transition=zoomin:"
            elif transition.kind is TransitionKind.MEMBER_BLUR_BRIDGE:
                xfade = "xfade=transition=hblur:"
            else:  # pragma: no cover - guarded by uses_overlap
                raise ValueError("unsupported overlap transition kind")
            parts.append(
                f"[{current_label}][v{next_unit.first_segment_index}]"
                f"{xfade}"
                f"duration={transition.duration_seconds(plan.fps):.9f}:"
                f"offset={cumulative_frames / plan.fps:.9f},format=yuv420p"
                f"[{output_label}]"
            )
            current_label = output_label
            cumulative_frames += next_unit.frame_count
            xfade_index += 1
            unit_index = next_unit_index
            unit = next_unit
        # xfade's EOF PTS can include one extra overlap endpoint frame. Bind
        # each chunk to its nominal frame budget before the next hard cut,
        # rather than correcting only the final concatenated video length.
        if chunk_had_overlap:
            chunk_label = f"vchunk{len(chunk_labels)}"
            parts.append(
                f"[{current_label}]trim=end_frame={cumulative_frames},"
                f"setpts=N/({plan.fps:.6f}*TB),settb=AVTB[{chunk_label}]"
            )
            current_label = chunk_label
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
    video_bitrate: str = "24M",
    video_maxrate: str = "36M",
    video_buffer_size: str = "48M",
    audio_bitrate: str = "320k",
    use_verified_frame_schedule: bool = True,
    progress_callback: Callable[[dict], None] | None = None,
    maximum_runtime_seconds: float | None = None,
) -> str:
    for label, value in (
        ("video_bitrate", video_bitrate),
        ("video_maxrate", video_maxrate),
        ("video_buffer_size", video_buffer_size),
        ("audio_bitrate", audio_bitrate),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a non-empty FFmpeg rate")
    if not isinstance(use_verified_frame_schedule, bool):
        raise TypeError("frame schedule mode must be a boolean")
    if maximum_runtime_seconds is not None and (
        isinstance(maximum_runtime_seconds, bool)
        or not isinstance(maximum_runtime_seconds, (int, float))
        or not math.isfinite(float(maximum_runtime_seconds))
        or maximum_runtime_seconds <= 0
    ):
        raise ValueError("maximum_runtime_seconds must be positive and finite")
    source_clocks: dict[int, VerifiedRenderFrameClock] = {}
    if use_verified_frame_schedule:
        for index in sorted({segment.source_index for segment in plan.segments}):
            clock = probe_render_frame_clock(
                plan.sources[index], ffprobe=str(Path(ffmpeg).with_name("ffprobe")),
            )
            if clock is not None:
                source_clocks[index] = clock
    filter_graph = build_filter_graph(plan, source_clocks=source_clocks)
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary output on the destination volume so the final replace
    # is atomic even when the user renders to an external SSD.
    with tempfile.TemporaryDirectory(
        prefix=".ive-crosscut-", dir=plan.output.parent
    ) as temp:
        root = Path(temp)
        filter_path = root / "filter.ffscript"
        filter_path.write_text(filter_graph, encoding="utf-8")
        clock_evidence = root / "frame-clock.json"
        clock_evidence.write_text(json.dumps({
            "contract": FRAME_SCHEDULE_CONTRACT,
            "enabled": use_verified_frame_schedule,
            "source_clocks": {str(index): clock.to_dict() for index, clock in source_clocks.items()},
            "timestamp_fallback_source_indices": sorted({segment.source_index for segment in plan.segments} - source_clocks.keys()),
            "plan": plan.to_json(), "filter_graph": filter_graph,
            "quality_approval": False,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
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
                        video_bitrate,
                        "-maxrate",
                        video_maxrate,
                        "-bufsize",
                        video_buffer_size,
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
                    audio_bitrate,
                    "-movflags",
                    "+faststart",
                    str(temporary_output),
                )
            )
            try:
                if progress_callback is None:
                    completed = subprocess.run(
                        command,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=maximum_runtime_seconds,
                    )
                else:
                    from .render_process import run_observed_render
                    completed = run_observed_render(
                        command,
                        stderr_path=root/'encoder-stderr.log',
                        progress_callback=progress_callback,
                        timeout_seconds=maximum_runtime_seconds,
                    )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "video render exceeded its bounded runtime: "
                    f"{maximum_runtime_seconds} seconds"
                ) from exc
            if completed.returncode == 0:
                temporary_output.replace(plan.output)
                clock_evidence.replace(plan.output.with_suffix(plan.output.suffix + ".frame-clock.json"))
                return codec
            failures.append(f"{codec}: {completed.stderr.strip() or 'unknown FFmpeg error'}")
        raise RuntimeError("video render failed: " + " | ".join(failures))


def write_plan(plan: RenderPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")


def contiguous_segments(segments: Iterable[RenderSegment]) -> tuple[RenderSegment, ...]:
    return tuple(sorted(segments, key=lambda segment: segment.master_start))
