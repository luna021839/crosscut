"""Dense, non-identifying editorial-camera review for official member fancams.

Automatic member sources are already authenticated as official target-member
9:16 fancams by metadata and media contracts.  This layer still never names a
face from pixels: it only uses a continuously dominant face box around a
selected hard cut, then re-decodes every transformed output frame before a
bounded scale/position match is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Sequence

import cv2

from .direct_member_visual import DirectMemberVisualSelection
from .direct_member_transform import (
    DirectMemberBoundaryTransformRepair,
    DirectMemberSourceFrameGeometry,
    direct_member_source_frame_geometry,
    map_box_to_portrait_fill,
)
from .editorial_motion import EditorialMotionPlan, EditorialMotionSegment
from .member_editorial_motion import (
    MemberEditorialMotionConfig,
    _State,
    _center,
    _frame_schedule,
    _mismatch,
    _project,
    _state_for_match,
    _track_for_states,
    _visible_fraction,
)
from .pose import SILHOUETTE_BACKEND
from .render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderSegment,
    RenderTransition,
    TransitionKind,
)
from .visual_score import describe_frame


_FACE_ATTENTION_BASIS = "unambiguous_dominant_face_bbox_geometry_only"


@dataclass(frozen=True, slots=True)
class DirectMemberFaceSample:
    bbox_xywh: tuple[float, float, float, float]
    face_count: int
    detector_backend: str
    subject_bbox_xywh: tuple[float, float, float, float] | None = None
    subject_confidence: float = 0.0
    subject_geometry_basis: str = "none"

    def __post_init__(self) -> None:
        if len(self.bbox_xywh) != 4:
            raise ValueError("direct member face box needs four values")
        x, y, width, height = tuple(float(value) for value in self.bbox_xywh)
        if any(not math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError("direct member face box must be finite")
        if not (
            0.0 <= x < 1.0
            and 0.0 <= y < 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
            and x + width <= 1.0 + 1e-7
            and y + height <= 1.0 + 1e-7
        ):
            raise ValueError("direct member face box must fit normalized bounds")
        if (
            isinstance(self.face_count, bool)
            or not isinstance(self.face_count, int)
            or self.face_count not in {0, 1}
        ):
            raise ValueError(
                "direct member geometry sample needs exactly one face or "
                "verified subject-only geometry"
            )
        backend = self.detector_backend.strip().casefold()
        subject_box = self.subject_bbox_xywh
        if subject_box is not None:
            if len(subject_box) != 4:
                raise ValueError("direct member subject box needs four values")
            sx, sy, swidth, sheight = tuple(float(value) for value in subject_box)
            if any(
                not math.isfinite(value)
                for value in (sx, sy, swidth, sheight)
            ) or not (
                0.0 <= sx < 1.0
                and 0.0 <= sy < 1.0
                and 0.0 < swidth <= 1.0
                and 0.0 < sheight <= 1.0
                and sx + swidth <= 1.0 + 1e-7
                and sy + sheight <= 1.0 + 1e-7
            ):
                raise ValueError(
                    "direct member subject box must fit normalized bounds"
                )
            subject_box = (sx, sy, swidth, sheight)
        confidence = float(self.subject_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("direct member subject confidence must be normalized")
        basis = self.subject_geometry_basis.strip()
        if subject_box is None:
            if confidence != 0.0 or basis != "none":
                raise ValueError(
                    "direct member missing subject box cannot claim geometry"
                )
        elif confidence <= 0.0 or basis not in {
            "pphumanseg_full_subject_silhouette",
            "face_associated_formation_person_box",
        }:
            raise ValueError("direct member subject geometry is unverified")
        if self.face_count == 1:
            if backend != "yunet":
                raise ValueError(
                    "direct member face sample needs verified YuNet geometry"
                )
        elif (
            backend != "none"
            or subject_box is None
            or any(
                abs(value - subject_value) > 1e-7
                for value, subject_value in zip(
                    (x, y, width, height),
                    subject_box,
                    strict=True,
                )
            )
        ):
            raise ValueError(
                "subject-only geometry must use its verified body box"
            )
        object.__setattr__(self, "bbox_xywh", (x, y, width, height))
        object.__setattr__(self, "detector_backend", backend)
        object.__setattr__(self, "subject_bbox_xywh", subject_box)
        object.__setattr__(self, "subject_confidence", confidence)
        object.__setattr__(self, "subject_geometry_basis", basis)

    @property
    def alignment_geometry_basis(self) -> str:
        if self.face_count == 1:
            return "unambiguous_dominant_face_bbox_geometry_only"
        if self.subject_geometry_basis == "pphumanseg_full_subject_silhouette":
            return "aggregate_human_silhouette"
        return "unambiguous_dominant_person_box_geometry_only"


DirectMemberGeometryProvider = Callable[
    [int, float], DirectMemberFaceSample | None
]


class DirectMemberDenseFaceProvider:
    """Decode only selected face samples; cache geometry, never pixels."""

    def __init__(self, sources: Sequence[Path]) -> None:
        self._sources = tuple(Path(source) for source in sources)
        self._captures: list[cv2.VideoCapture | None] = []
        self._cache: dict[
            tuple[int, int], DirectMemberFaceSample | None
        ] = {}
        self._request_count = 0
        self._cache_hit_count = 0
        self._decode_count = 0
        for source in self._sources:
            capture = cv2.VideoCapture(str(source))
            if capture.isOpened():
                self._captures.append(capture)
            else:
                capture.release()
                self._captures.append(None)

    def sample(
        self,
        source_index: int,
        source_local_seconds: float,
    ) -> DirectMemberFaceSample | None:
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or not 0 <= source_index < len(self._sources)
        ):
            raise ValueError("dense member face source index is invalid")
        if not math.isfinite(source_local_seconds):
            raise ValueError("dense member face time must be finite")
        self._request_count += 1
        key = (source_index, int(round(max(0.0, source_local_seconds) * 1000.0)))
        if key in self._cache:
            self._cache_hit_count += 1
            return self._cache[key]
        capture = self._captures[source_index]
        if capture is None:
            self._cache[key] = None
            return None
        self._decode_count += 1
        capture.set(cv2.CAP_PROP_POS_MSEC, key[1])
        success, frame = capture.read()
        if not success or frame is None:
            self._cache[key] = None
            return None
        try:
            descriptor = describe_frame(
                frame,
                include_subject_geometry=True,
                include_formation_pose=True,
            )
            subject_box = None
            subject_confidence = 0.0
            subject_basis = "none"
            if (
                descriptor.silhouette_bbox_xywh is not None
                and descriptor.silhouette_confidence >= 0.30
                and descriptor.silhouette_backend == SILHOUETTE_BACKEND
            ):
                subject_box = descriptor.silhouette_bbox_xywh
                subject_confidence = descriptor.silhouette_confidence
                subject_basis = "pphumanseg_full_subject_silhouette"
            elif descriptor.primary_face_bbox is not None:
                face = descriptor.primary_face_bbox
                face_center = (
                    face[0] + face[2] / 2.0,
                    face[1] + face[3] / 2.0,
                )
                associated = tuple(
                    person
                    for person in descriptor.formation_people
                    if person.bbox_xywh[0] - 1e-6
                    <= face_center[0]
                    <= person.bbox_xywh[0] + person.bbox_xywh[2] + 1e-6
                    and person.bbox_xywh[1] - 1e-6
                    <= face_center[1]
                    <= person.bbox_xywh[1] + person.bbox_xywh[3] + 1e-6
                    and person.detection_confidence >= 0.50
                )
                if len(associated) == 1:
                    subject_box = associated[0].bbox_xywh
                    subject_confidence = associated[0].detection_confidence
                    subject_basis = "face_associated_formation_person_box"
            if (
                descriptor.primary_face_bbox is not None
                and descriptor.face_count == 1
                and descriptor.face_detector_backend.casefold() == "yunet"
            ):
                sample = DirectMemberFaceSample(
                    descriptor.primary_face_bbox,
                    descriptor.face_count,
                    descriptor.face_detector_backend,
                    subject_box,
                    subject_confidence,
                    subject_basis,
                )
            elif subject_box is not None:
                sample = DirectMemberFaceSample(
                    subject_box,
                    0,
                    "none",
                    subject_box,
                    subject_confidence,
                    subject_basis,
                )
            else:
                sample = None
        except (cv2.error, ValueError, FloatingPointError):
            sample = None
        self._cache[key] = sample
        return sample

    def diagnostics(self) -> dict[str, int | bool | str]:
        return {
            "backend": (
                "yunet_face_or_verified_full_subject_geometry_"
                "dense_selected_ramps"
            ),
            "request_count": self._request_count,
            "cache_hit_count": self._cache_hit_count,
            "decode_count": self._decode_count,
            "cache_entry_count": len(self._cache),
            "pixels_persisted": False,
            "face_embeddings_computed": False,
        }

    def close(self) -> None:
        for capture in self._captures:
            if capture is not None:
                capture.release()


@dataclass(frozen=True, slots=True)
class DirectMemberEditorialMotionConfig:
    ramp_seconds: float = 0.75
    maximum_boundary_zoom: float = 1.35
    maximum_center_shift: float = 0.080
    maximum_boundary_center_distance: float = 0.30
    maximum_residual_center_distance: float = 0.08
    minimum_mismatch_improvement_fraction: float = 0.20
    minimum_face_visible_fraction: float = 0.995
    legacy_face_only_maximum_zoom: float = 1.12
    legacy_face_only_maximum_center_shift: float = 0.025
    minimum_widened_subject_confidence: float = 0.30
    minimum_widened_subject_visible_fraction: float = 0.995
    maximum_dense_subject_center_step: float = 0.12
    maximum_dense_subject_height_ratio: float = 1.40
    maximum_dense_center_step: float = 0.10
    maximum_dense_height_ratio: float = 1.35
    maximum_endpoint_center_disagreement: float = 0.08
    maximum_endpoint_height_ratio: float = 1.25
    target_events_per_minute: float = 12.0
    minimum_event_spacing_seconds: float = 2.0
    enable_interior_zoom_pulses: bool = False
    target_rich_motion_events_per_minute: float = 4.0
    minimum_interior_pulse_duration_seconds: float = 2.2
    interior_pulse_zoom: float = 1.05

    def __post_init__(self) -> None:
        values = tuple(float(getattr(self, name)) for name in (
            "ramp_seconds",
            "maximum_boundary_zoom",
            "maximum_center_shift",
            "maximum_boundary_center_distance",
            "maximum_residual_center_distance",
            "minimum_mismatch_improvement_fraction",
            "minimum_face_visible_fraction",
            "legacy_face_only_maximum_zoom",
            "legacy_face_only_maximum_center_shift",
            "minimum_widened_subject_confidence",
            "minimum_widened_subject_visible_fraction",
            "maximum_dense_subject_center_step",
            "maximum_dense_subject_height_ratio",
            "maximum_dense_center_step",
            "maximum_dense_height_ratio",
            "maximum_endpoint_center_disagreement",
            "maximum_endpoint_height_ratio",
            "target_events_per_minute",
            "minimum_event_spacing_seconds",
            "target_rich_motion_events_per_minute",
            "minimum_interior_pulse_duration_seconds",
            "interior_pulse_zoom",
        ))
        if any(not math.isfinite(value) for value in values):
            raise ValueError("direct member editorial configuration must be finite")
        # Reuse the confirmed-member configuration as the shared camera/schema
        # contract.  Dense-direct adds stricter continuity gates below.
        _shared_settings(self)
        if not 0.0 <= self.maximum_dense_center_step <= 0.20:
            raise ValueError("dense face center-step gate is invalid")
        if not 1.0 <= self.maximum_dense_height_ratio <= 2.0:
            raise ValueError("dense face height-ratio gate is invalid")
        if not 1.0 <= self.legacy_face_only_maximum_zoom <= 1.12:
            raise ValueError("legacy face-only zoom boundary is invalid")
        if not 0.0 <= self.legacy_face_only_maximum_center_shift <= 0.025:
            raise ValueError("legacy face-only pan boundary is invalid")
        if not 0.0 <= self.minimum_widened_subject_confidence <= 1.0:
            raise ValueError("widened subject confidence gate is invalid")
        if not 0.95 <= self.minimum_widened_subject_visible_fraction <= 1.0:
            raise ValueError("widened subject containment gate is invalid")
        if not 0.0 <= self.maximum_dense_subject_center_step <= 0.20:
            raise ValueError("dense subject center-step gate is invalid")
        if not 1.0 <= self.maximum_dense_subject_height_ratio <= 2.0:
            raise ValueError("dense subject height-ratio gate is invalid")
        if not 0.0 <= self.maximum_endpoint_center_disagreement <= 0.20:
            raise ValueError("dense endpoint center gate is invalid")
        if not 1.0 <= self.maximum_endpoint_height_ratio <= 2.0:
            raise ValueError("dense endpoint height-ratio gate is invalid")
        if not isinstance(self.enable_interior_zoom_pulses, bool):
            raise ValueError("interior zoom pulse switch must be boolean")
        if not 0.0 <= self.target_rich_motion_events_per_minute <= 12.0:
            raise ValueError("rich direct-member motion density is invalid")
        if not 1.5 <= self.minimum_interior_pulse_duration_seconds <= 6.0:
            raise ValueError("interior zoom pulse duration floor is invalid")
        if not 1.02 <= self.interior_pulse_zoom <= 1.12:
            raise ValueError("interior zoom pulse must remain subtle")


def _shared_settings(
    config: DirectMemberEditorialMotionConfig,
) -> MemberEditorialMotionConfig:
    return MemberEditorialMotionConfig(
        ramp_seconds=config.ramp_seconds,
        maximum_boundary_zoom=config.maximum_boundary_zoom,
        maximum_center_shift=config.maximum_center_shift,
        maximum_boundary_center_distance=config.maximum_boundary_center_distance,
        maximum_residual_center_distance=config.maximum_residual_center_distance,
        minimum_mismatch_improvement_fraction=(
            config.minimum_mismatch_improvement_fraction
        ),
        minimum_subject_visible_fraction=config.minimum_face_visible_fraction,
        target_events_per_minute=config.target_events_per_minute,
        minimum_event_spacing_seconds=config.minimum_event_spacing_seconds,
    )


@dataclass(frozen=True, slots=True)
class _BoundaryCandidate:
    boundary_index: int
    event_time: float
    score: float
    side: str
    state: _State
    alternative_states: tuple[_State, ...] = ()
    required: bool = False


def _attention_alignment_geometry(
    sample: object,
) -> tuple[tuple[float, float, float, float], str] | None:
    attention = getattr(sample, "attention", None)
    if attention is None:
        return None
    basis = str(getattr(attention, "basis", ""))
    descriptor = getattr(sample, "descriptor", None)
    if descriptor is None:
        return None
    if basis == _FACE_ATTENTION_BASIS:
        if getattr(descriptor, "face_count", None) != 1:
            return None
    elif basis == "aggregate_human_silhouette":
        if (
            getattr(descriptor, "silhouette_backend", None)
            != SILHOUETTE_BACKEND
            or float(getattr(descriptor, "silhouette_confidence", 0.0))
            < 0.30
        ):
            return None
    elif basis == "unambiguous_dominant_person_box_geometry_only":
        if not tuple(getattr(descriptor, "formation_people", ())):
            return None
    else:
        return None
    raw = getattr(attention, "bbox_xywh", None)
    if not isinstance(raw, tuple) or len(raw) != 4:
        return None
    box = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) for value in box):
        return None
    return (box, basis)  # type: ignore[return-value]


def _endpoint_agrees(
    dense_box: tuple[float, float, float, float],
    audited_box: tuple[float, float, float, float],
    config: DirectMemberEditorialMotionConfig,
) -> bool:
    center_distance = math.hypot(
        _center(dense_box)[0] - _center(audited_box)[0],
        _center(dense_box)[1] - _center(audited_box)[1],
    )
    height_ratio = max(dense_box[3], audited_box[3]) / max(
        1e-9, min(dense_box[3], audited_box[3])
    )
    return (
        center_distance <= config.maximum_endpoint_center_disagreement + 1e-9
        and height_ratio <= config.maximum_endpoint_height_ratio + 1e-9
    )


def _neutral(track: EditorialTransformTrack, time_seconds: float) -> bool:
    state = track.evaluate(time_seconds)
    return (
        abs(state.zoom - 1.0) <= 1e-7
        and abs(state.center_x - 0.5) <= 1e-7
        and abs(state.center_y - 0.5) <= 1e-7
        and abs(state.roll_degrees) <= 1e-7
    )


def _track_is_dense_face_safe(
    *,
    segment_index: int,
    eligible_source_index: int,
    segment: RenderSegment,
    track: EditorialTransformTrack,
    frame_count: int,
    common_master_start: float,
    geometry_provider: DirectMemberGeometryProvider,
    fps: float,
    config: DirectMemberEditorialMotionConfig,
    source_frame_geometry: DirectMemberSourceFrameGeometry | None,
) -> bool:
    active_frames = tuple(
        not _neutral(track, frame_index / fps)
        for frame_index in range(frame_count)
    )
    # Include one neutral-frame halo on each side of every transformed run.
    # Otherwise a detector/subject jump exactly where the ramp begins or ends
    # has no adjacent dense sample and can look continuous inside the ramp.
    review_frames: set[int] = {
        frame_index
        for frame_index, active in enumerate(active_frames)
        if active
    }
    for frame_index in tuple(review_frames):
        if frame_index > 0:
            review_frames.add(frame_index - 1)
        if frame_index + 1 < frame_count:
            review_frames.add(frame_index + 1)

    widened_subject_proof_required = bool(
        any(
            keyframe.zoom
            > config.legacy_face_only_maximum_zoom + 1e-9
            or abs(keyframe.center_x - 0.5)
            > config.legacy_face_only_maximum_center_shift + 1e-9
            or abs(keyframe.center_y - 0.5)
            > config.legacy_face_only_maximum_center_shift + 1e-9
            for keyframe in track.keyframes
        )
    )

    previous_box: tuple[float, float, float, float] | None = None
    previous_subject_box: tuple[float, float, float, float] | None = None
    previous_frame_index: int | None = None
    for frame_index in sorted(review_frames):
        local_time = frame_index / fps
        if (
            previous_frame_index is None
            or frame_index != previous_frame_index + 1
        ):
            previous_box = None
            previous_subject_box = None
        canonical_time = common_master_start + segment.master_start + local_time
        sample = geometry_provider(eligible_source_index, canonical_time)
        if sample is None:
            return False
        box = (
            map_box_to_portrait_fill(
                sample.bbox_xywh,
                source_frame_geometry,
            )
            if source_frame_geometry is not None
            else sample.bbox_xywh
        )
        if previous_box is not None:
            center_step = math.hypot(
                _center(box)[0] - _center(previous_box)[0],
                _center(box)[1] - _center(previous_box)[1],
            )
            height_ratio = max(box[3], previous_box[3]) / max(
                1e-9, min(box[3], previous_box[3])
            )
            if (
                center_step > config.maximum_dense_center_step + 1e-9
                or height_ratio > config.maximum_dense_height_ratio + 1e-9
            ):
                return False
        state_sample = track.evaluate(local_time)
        state = _State(
            state_sample.zoom,
            state_sample.center_x,
            state_sample.center_y,
        )
        if (
            _visible_fraction(box, state) + 1e-9
            < config.minimum_face_visible_fraction
        ):
            return False
        if widened_subject_proof_required:
            subject_box = sample.subject_bbox_xywh
            if (
                subject_box is None
                or sample.subject_confidence + 1e-9
                < config.minimum_widened_subject_confidence
                or sample.subject_geometry_basis
                not in {
                    "pphumanseg_full_subject_silhouette",
                    "face_associated_formation_person_box",
                }
            ):
                return False
            if source_frame_geometry is not None:
                subject_box = map_box_to_portrait_fill(
                    subject_box,
                    source_frame_geometry,
                )
            if previous_subject_box is not None:
                subject_center_step = math.hypot(
                    _center(subject_box)[0] - _center(previous_subject_box)[0],
                    _center(subject_box)[1] - _center(previous_subject_box)[1],
                )
                subject_height_ratio = max(
                    subject_box[3], previous_subject_box[3]
                ) / max(1e-9, min(subject_box[3], previous_subject_box[3]))
                if (
                    subject_center_step
                    > config.maximum_dense_subject_center_step + 1e-9
                    or subject_height_ratio
                    > config.maximum_dense_subject_height_ratio + 1e-9
                ):
                    return False
            if (
                _visible_fraction(subject_box, state) + 1e-9
                < config.minimum_widened_subject_visible_fraction
            ):
                return False
            previous_subject_box = subject_box
        previous_box = box
        previous_frame_index = frame_index
    return True


def _resolved_source_frame_geometries(
    visual_selection: DirectMemberVisualSelection,
    *,
    source_count: int,
    source_frame_sizes: Sequence[tuple[int, int]] | None,
    output_frame_size: tuple[int, int],
) -> tuple[DirectMemberSourceFrameGeometry | None, ...]:
    stored = tuple(
        getattr(visual_selection, "source_frame_geometries", ())
    )
    if stored:
        if len(stored) != source_count or any(
            item is not None
            and not isinstance(item, DirectMemberSourceFrameGeometry)
            for item in stored
        ):
            raise ValueError("direct member stored source geometry is invalid")
        return stored
    if source_frame_sizes is None:
        return (None,) * source_count
    if len(source_frame_sizes) != source_count:
        raise ValueError("direct member source frame sizes must match sources")
    return tuple(
        direct_member_source_frame_geometry(
            size,
            output_size=output_frame_size,
        )
        for size in source_frame_sizes
    )


def build_direct_member_editorial_motion_plan(
    *,
    visual_selection: DirectMemberVisualSelection,
    render_segments: Sequence[RenderSegment],
    transitions: Sequence[RenderTransition],
    common_master_start: float,
    geometry_provider: DirectMemberGeometryProvider,
    fps: float = 29.97,
    config: DirectMemberEditorialMotionConfig | None = None,
    source_frame_sizes: Sequence[tuple[int, int]] | None = None,
    output_frame_size: tuple[int, int] = (1080, 1920),
) -> EditorialMotionPlan:
    """Add dense-verified face or full-subject matches to direct 9:16 cuts."""

    renders = tuple(render_segments)
    transition_items = tuple(transitions)
    source_path = tuple(int(value) for value in visual_selection.source_indices)
    boundaries = tuple(visual_selection.boundaries)
    if not renders or len(renders) != len(source_path):
        raise ValueError("direct member motion needs one selected source per segment")
    if len(transition_items) != len(renders) - 1 or len(boundaries) != len(renders) - 1:
        raise ValueError("direct member motion needs one audited transition per boundary")
    if not math.isfinite(common_master_start) or common_master_start < 0.0:
        raise ValueError("direct member common master start is invalid")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("direct member editorial fps is invalid")
    settings = config or DirectMemberEditorialMotionConfig()
    shared = _shared_settings(settings)
    source_count = max(source_path) + 1
    source_frame_geometries = _resolved_source_frame_geometries(
        visual_selection,
        source_count=source_count,
        source_frame_sizes=source_frame_sizes,
        output_frame_size=output_frame_size,
    )
    frame_counts, last_frame_times, boundary_times = _frame_schedule(renders, fps)
    if any(segment.crop is not None or segment.pan_crop is not None for segment in renders):
        raise ValueError("direct member editorial motion requires original full frames")

    blocked: set[int] = set()
    for index, transition in enumerate(transition_items):
        if (
            transition.kind.uses_overlap
            or transition.tier == "D"
        ):
            blocked.update((index, index + 1))
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    candidates: list[_BoundaryCandidate] = []
    required_boundary_indices: set[int] = set()
    for boundary_index, diagnostic in enumerate(boundaries):
        if getattr(diagnostic, "boundary_index", None) != boundary_index:
            raise ValueError("direct member boundary indices must be contiguous")
        left_source = source_path[boundary_index]
        right_source = source_path[boundary_index + 1]
        if left_source == right_source:
            continue
        mismatch = getattr(diagnostic, "mismatch", None)
        repair = getattr(mismatch, "transform_repair", None)
        if repair is not None and not isinstance(
            repair,
            DirectMemberBoundaryTransformRepair,
        ):
            raise ValueError("direct member boundary transform repair is invalid")
        required = repair is not None
        if required:
            required_boundary_indices.add(boundary_index)
        if boundary_index in blocked or boundary_index + 1 in blocked:
            skip("dissolve_adjacent")
            continue
        if (
            getattr(diagnostic, "outgoing_eligible_source_index", None)
            != left_source
            or getattr(diagnostic, "incoming_eligible_source_index", None)
            != right_source
        ):
            raise ValueError("direct member boundary source evidence disagrees with path")
        audited_left_geometry = _attention_alignment_geometry(
            diagnostic.outgoing_sample
        )
        audited_right_geometry = _attention_alignment_geometry(
            diagnostic.incoming_sample
        )
        if (
            audited_left_geometry is None
            or audited_right_geometry is None
            or audited_left_geometry[1] != audited_right_geometry[1]
        ):
            skip("no_unambiguous_face_pair")
            continue
        audited_left, audited_basis = audited_left_geometry
        audited_right, _right_basis = audited_right_geometry
        left_geometry = source_frame_geometries[left_source]
        right_geometry = source_frame_geometries[right_source]
        audited_left = (
            map_box_to_portrait_fill(audited_left, left_geometry)
            if left_geometry is not None
            else audited_left
        )
        audited_right = (
            map_box_to_portrait_fill(audited_right, right_geometry)
            if right_geometry is not None
            else audited_right
        )
        left_time = (
            common_master_start
            + renders[boundary_index].master_start
            + last_frame_times[boundary_index]
        )
        right_time = common_master_start + renders[boundary_index + 1].master_start
        left_sample = geometry_provider(left_source, left_time)
        right_sample = geometry_provider(right_source, right_time)
        if left_sample is None or right_sample is None:
            skip("dense_boundary_face_missing")
            continue
        if (
            left_sample.alignment_geometry_basis != audited_basis
            or right_sample.alignment_geometry_basis != audited_basis
        ):
            skip("dense_boundary_face_disagrees_with_audit")
            continue
        left_box = (
            map_box_to_portrait_fill(left_sample.bbox_xywh, left_geometry)
            if left_geometry is not None
            else left_sample.bbox_xywh
        )
        right_box = (
            map_box_to_portrait_fill(right_sample.bbox_xywh, right_geometry)
            if right_geometry is not None
            else right_sample.bbox_xywh
        )
        if not _endpoint_agrees(left_box, audited_left, settings) or not _endpoint_agrees(
            right_box, audited_right, settings
        ):
            skip("dense_boundary_face_disagrees_with_audit")
            continue
        center_distance = math.hypot(
            _center(left_box)[0] - _center(right_box)[0],
            _center(left_box)[1] - _center(right_box)[1],
        )
        if center_distance > settings.maximum_boundary_center_distance + 1e-9:
            skip("boundary_position_exceeds_transform_bound")
            continue
        ratio = max(left_box[3], right_box[3]) / max(
            1e-9, min(left_box[3], right_box[3])
        )
        if ratio <= 1.015 and center_distance <= 0.025:
            skip("already_scale_position_matched")
            continue
        baseline = _mismatch(left_box, right_box)
        if required and repair is not None:
            if repair.alignment_geometry_basis != audited_basis:
                raise ValueError(
                    "required transform geometry basis disagrees with dense audit"
                )
            side = "left" if repair.transformed_side == "outgoing" else "right"
            affected_index = (
                boundary_index if side == "left" else boundary_index + 1
            )
            box, target_box = (
                (left_box, right_box) if side == "left" else (right_box, left_box)
            )
            expected_source = (
                left_source if side == "left" else right_source
            )
            if repair.transformed_source_index != expected_source:
                raise ValueError(
                    "direct member required transform source disagrees with path"
                )
        elif left_box[3] + 1e-9 < right_box[3]:
            side = "left"
            affected_index = boundary_index
            box, target_box = left_box, right_box
        else:
            side = "right"
            affected_index = boundary_index + 1
            box, target_box = right_box, left_box
        if last_frame_times[affected_index] < 0.55 - 1e-9:
            skip("window_shorter_than_reference_ramp")
            continue
        affected_source = source_path[affected_index]
        affected_geometry = source_frame_geometries[affected_source]
        lossless_zoom_limit = (
            min(
                settings.maximum_boundary_zoom,
                affected_geometry.lossless_zoom_capacity,
            )
            if affected_geometry is not None
            else settings.maximum_boundary_zoom
        )
        if required and affected_geometry is None:
            skip("required_transform_source_resolution_missing")
            continue
        if lossless_zoom_limit <= 1.0 + 1e-9:
            skip("source_resolution_has_no_lossless_transform_headroom")
            continue
        requested_zoom = max(1.02, target_box[3] / box[3])
        zoom = min(lossless_zoom_limit, requested_zoom)
        if required and repair is not None:
            zoom = min(lossless_zoom_limit, max(zoom, repair.zoom))
        qualifying: list[tuple[float, _State]] = []
        for state in _state_for_match(
            box=box,
            target_box=target_box,
            zoom=zoom,
            settings=shared,
        ):
            if (
                _visible_fraction(box, state) + 1e-9
                < settings.minimum_face_visible_fraction
            ):
                continue
            projected_left = _project(left_box, state) if side == "left" else left_box
            projected_right = _project(right_box, state) if side == "right" else right_box
            transformed = _mismatch(projected_left, projected_right)
            residual_center = math.hypot(
                _center(projected_left)[0] - _center(projected_right)[0],
                _center(projected_left)[1] - _center(projected_right)[1],
            )
            if residual_center > settings.maximum_residual_center_distance + 1e-9:
                continue
            if transformed > baseline * (
                1.0 - settings.minimum_mismatch_improvement_fraction
            ) + 1e-9:
                continue
            if transformed + 1e-4 < baseline:
                qualifying.append((transformed, state))
        if not qualifying:
            skip("no_improving_contained_transform")
            continue
        qualifying.sort(key=lambda item: item[0])
        best_score, best_state = qualifying[0]
        candidates.append(
            _BoundaryCandidate(
                boundary_index=boundary_index,
                event_time=boundary_times[boundary_index],
                score=baseline - best_score,
                side=side,
                state=best_state,
                alternative_states=tuple(state for _score, state in qualifying[1:]),
                required=required,
            )
        )

    quantized_duration = sum(frame_counts) / fps
    maximum_events = int(math.floor(
        quantized_duration * settings.target_events_per_minute / 60.0 + 1e-12
    ))
    entry_states: list[_State | None] = [None] * len(renders)
    exit_states: list[_State | None] = [None] * len(renders)
    tracks: list[EditorialTransformTrack | None] = [None] * len(renders)
    accepted: list[_BoundaryCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            not item.required,
            -item.score,
            item.event_time,
        ),
    ):
        if not candidate.required and len(accepted) >= maximum_events:
            skip("event_density_cap")
            continue
        if not candidate.required and any(
            abs(candidate.event_time - item.event_time)
            < settings.minimum_event_spacing_seconds - 1e-9
            for item in accepted
        ):
            skip("event_spacing_cap")
            continue
        affected_index = (
            candidate.boundary_index
            if candidate.side == "left"
            else candidate.boundary_index + 1
        )
        previous_entry = entry_states[affected_index]
        previous_exit = exit_states[affected_index]
        accepted_state: _State | None = None
        accepted_track: EditorialTransformTrack | None = None
        for state in (candidate.state,) + candidate.alternative_states:
            if candidate.side == "left":
                exit_states[affected_index] = state
            else:
                entry_states[affected_index] = state
            try:
                track = _track_for_states(
                    entry=entry_states[affected_index],
                    exit=exit_states[affected_index],
                    last_frame_time=last_frame_times[affected_index],
                    settings=shared,
                )
            except ValueError:
                track = None
            if track is not None and _track_is_dense_face_safe(
                segment_index=affected_index,
                eligible_source_index=source_path[affected_index],
                segment=renders[affected_index],
                track=track,
                frame_count=frame_counts[affected_index],
                common_master_start=common_master_start,
                geometry_provider=geometry_provider,
                fps=fps,
                config=settings,
                source_frame_geometry=source_frame_geometries[
                    source_path[affected_index]
                ],
            ):
                accepted_state = state
                accepted_track = track
                break
        if accepted_state is None or accepted_track is None:
            entry_states[affected_index] = previous_entry
            exit_states[affected_index] = previous_exit
            skip("dense_face_containment_or_continuity_failed")
            continue
        tracks[affected_index] = accepted_track
        accepted.append(
            _BoundaryCandidate(
                boundary_index=candidate.boundary_index,
                event_time=candidate.event_time,
                score=candidate.score,
                side=candidate.side,
                state=accepted_state,
                required=candidate.required,
            )
        )

    interior_pulse_segments: set[int] = set()
    if settings.enable_interior_zoom_pulses:
        target_rich_motion_count = min(
            maximum_events,
            math.ceil(
                quantized_duration
                * settings.target_rich_motion_events_per_minute
                / 60.0
                - 1e-12
            ),
        )
        accepted_event_times = [item.event_time for item in accepted]

        def interior_activity_score(segment_index: int) -> float:
            values: list[float] = []
            for boundary_index in (segment_index - 1, segment_index):
                if not 0 <= boundary_index < len(boundaries):
                    continue
                motion = getattr(
                    getattr(boundaries[boundary_index], "mismatch", None),
                    "motion_connection",
                    None,
                )
                if motion is None:
                    continue
                for name in (
                    "outgoing_motion_magnitude",
                    "incoming_motion_magnitude",
                ):
                    value = getattr(motion, name, None)
                    if value is not None and math.isfinite(float(value)):
                        values.append(float(value))
            return max(values, default=0.0)

        pulse_candidates = sorted(
            range(len(renders)),
            key=lambda index: (
                -interior_activity_score(index),
                -last_frame_times[index],
                renders[index].master_start,
                index,
            ),
        )
        for index in pulse_candidates:
            if len(accepted) + len(interior_pulse_segments) >= target_rich_motion_count:
                break
            if tracks[index] is not None or index in blocked:
                continue
            if (
                last_frame_times[index]
                < settings.minimum_interior_pulse_duration_seconds - 1e-9
            ):
                skip("interior_pulse_segment_too_short")
                continue
            event_time = renders[index].master_start + last_frame_times[index] / 2.0
            if any(
                abs(event_time - other_time)
                < settings.minimum_event_spacing_seconds - 1e-9
                for other_time in accepted_event_times
            ):
                skip("interior_pulse_spacing_cap")
                continue
            source_geometry = source_frame_geometries[source_path[index]]
            if source_geometry is None:
                skip("interior_pulse_source_resolution_missing")
                continue
            zoom = min(
                settings.interior_pulse_zoom,
                settings.maximum_boundary_zoom,
                source_geometry.lossless_zoom_capacity,
            )
            if zoom < 1.02 - 1e-9:
                skip("interior_pulse_no_lossless_zoom_headroom")
                continue
            track = EditorialTransformTrack(
                keyframes=(
                    EditorialTransformKeyframe(0.0, 1.0),
                    EditorialTransformKeyframe(
                        last_frame_times[index] / 2.0,
                        zoom,
                    ),
                    EditorialTransformKeyframe(
                        last_frame_times[index],
                        1.0,
                    ),
                ),
                reason=(
                    "official_fancam_dense_subject_lossless_interior_zoom_pulse"
                ),
            )
            if not _track_is_dense_face_safe(
                segment_index=index,
                eligible_source_index=source_path[index],
                segment=renders[index],
                track=track,
                frame_count=frame_counts[index],
                common_master_start=common_master_start,
                geometry_provider=geometry_provider,
                fps=fps,
                config=settings,
                source_frame_geometry=source_geometry,
            ):
                skip("interior_pulse_dense_containment_or_continuity_failed")
                continue
            tracks[index] = track
            interior_pulse_segments.add(index)
            accepted_event_times.append(event_time)

    for index in range(len(renders)):
        if entry_states[index] is None and exit_states[index] is None:
            continue
        track = _track_for_states(
            entry=entry_states[index],
            exit=exit_states[index],
            last_frame_time=last_frame_times[index],
            settings=shared,
        )
        if track is None or not _track_is_dense_face_safe(
            segment_index=index,
            eligible_source_index=source_path[index],
            segment=renders[index],
            track=track,
            frame_count=frame_counts[index],
            common_master_start=common_master_start,
            geometry_provider=geometry_provider,
            fps=fps,
            config=settings,
            source_frame_geometry=source_frame_geometries[source_path[index]],
        ):
            raise ValueError("accepted direct member camera failed final dense review")
        tracks[index] = track

    accepted_segments = {
        item.boundary_index if item.side == "left" else item.boundary_index + 1
        for item in accepted
    }
    accepted_boundary_indices = tuple(
        sorted(item.boundary_index for item in accepted)
    )
    missing_required = tuple(
        sorted(required_boundary_indices.difference(accepted_boundary_indices))
    )
    if missing_required:
        raise ValueError(
            "required direct member transform failed dense review at boundaries "
            + ",".join(str(index) for index in missing_required)
        )
    return EditorialMotionPlan(
        segments=tuple(
            EditorialMotionSegment(
                segment_index=index,
                source_index=source_path[index],
                track=tracks[index],
                reasons=(
                    (
                        "official_fancam_dense_face_subject_"
                        "scale_position_match",
                    )
                    if index in accepted_segments
                    else (
                        "official_fancam_dense_subject_"
                        "lossless_interior_zoom_pulse",
                    )
                    if index in interior_pulse_segments
                    else ("no_safe_dense_face_editorial_motion",)
                ),
            )
            for index in range(len(renders))
        ),
        boundary_match_count=len(accepted),
        zoom_event_count=len(accepted) + len(interior_pulse_segments),
        pan_event_count=sum(
            abs(item.state.center_x - 0.5) > 1e-6
            or abs(item.state.center_y - 0.5) > 1e-6
            for item in accepted
        ),
        roll_event_count=0,
        interior_zoom_pulse_count=len(interior_pulse_segments),
        blocked_dissolve_segment_count=len(blocked),
        skipped_boundary_reasons=tuple(sorted(skipped.items())),
        analysis_basis=(
            "official_target_fancam_face_or_verified_full_subject_dense_"
            "output_frame_resolution_bound_repair_and_lossless_pulse_review_v5"
        ),
        accepted_boundary_indices=accepted_boundary_indices,
    )


__all__ = [
    "DirectMemberDenseFaceProvider",
    "DirectMemberEditorialMotionConfig",
    "DirectMemberFaceSample",
    "build_direct_member_editorial_motion_plan",
]
