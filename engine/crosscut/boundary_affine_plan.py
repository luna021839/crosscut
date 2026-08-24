"""Decode and review real source frames for affine-aware hard cuts.

``boundary_affine_match`` deliberately knows nothing about timelines or video
files.  This module supplies that missing production glue: it resolves the
exact outgoing/incoming source clocks, canonicalizes both frames to the same
16:9 output space, carries subject boxes through the same center crop, and
records a fail-closed decision for every real source switch.

The affine layer is intentionally disabled when either side already has a
base ``PanCropTrack``.  Estimating an affine in raw-source coordinates and
then applying it after a different crop would be mathematically dishonest.
The decoded pixels are never serialized.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from .boundary_affine_match import (
    BoundaryAffineMatchResult,
    BoundaryAffineTransformEvidence,
    match_boundary_affine,
)
from .face_match import FaceMatchPlan
from .render import RenderTransition, TransitionKind
from .visual_score import VisualTimelineChoice, VisualTimelinePlan


ANALYSIS_BASIS = (
    "exact_source_clock_canonical_output_boundary_subject_affine_review_v2"
)
FrameProvider = Callable[[int, float], np.ndarray | None]


@dataclass(frozen=True, slots=True)
class BoundaryFrameSubjectGeometry:
    """Non-identifying geometry measured in canonical output space.

    The visual optimizer's three samples are intentionally cheap and can sit a
    frame or two away from the actual edit.  This small payload is instead
    measured from the decoded last outgoing / first incoming frame.  It keeps
    only normalized boxes and confidence; pixels and appearance descriptors
    remain transient.
    """

    face_count: int = 0
    face_box_xywh: tuple[float, float, float, float] | None = None
    silhouette_box_xywh: tuple[float, float, float, float] | None = None
    silhouette_confidence: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.face_count, bool)
            or not isinstance(self.face_count, int)
            or self.face_count < 0
        ):
            raise ValueError("boundary face count must be a non-negative integer")
        for name in ("face_box_xywh", "silhouette_box_xywh"):
            box = getattr(self, name)
            if box is None:
                continue
            if len(box) != 4:
                raise ValueError(f"{name} needs four normalized values")
            x, y, box_width, box_height = tuple(float(value) for value in box)
            if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
                raise ValueError(f"{name} must be finite")
            if not (
                0.0 <= x <= 1.0
                and 0.0 <= y <= 1.0
                and 0.0 < box_width <= 1.0
                and 0.0 < box_height <= 1.0
                and x + box_width <= 1.000001
                and y + box_height <= 1.000001
            ):
                raise ValueError(f"{name} must fit normalized output bounds")
            object.__setattr__(self, name, (x, y, box_width, box_height))
        confidence = float(self.silhouette_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("silhouette confidence must be normalized")
        object.__setattr__(self, "silhouette_confidence", confidence)


@dataclass(frozen=True, slots=True)
class BoundarySubjectPair:
    """A common subject measured on the exact two rendered cut endpoints."""

    kind: str
    outgoing_box_xywh: tuple[float, float, float, float]
    incoming_box_xywh: tuple[float, float, float, float]
    analysis_basis: str = "exact_decoded_canonical_output_cut_frame_geometry_v1"

    def __post_init__(self) -> None:
        if self.kind not in {"face", "silhouette"}:
            raise ValueError("boundary subject pair kind is invalid")
        # Reuse the strict output-space validator without retaining redundant
        # detector metadata in the serialized plan.
        if self.kind == "face":
            BoundaryFrameSubjectGeometry(
                face_count=1,
                face_box_xywh=self.outgoing_box_xywh,
            )
            BoundaryFrameSubjectGeometry(
                face_count=1,
                face_box_xywh=self.incoming_box_xywh,
            )
        else:
            BoundaryFrameSubjectGeometry(
                silhouette_box_xywh=self.outgoing_box_xywh,
                silhouette_confidence=1.0,
            )
            BoundaryFrameSubjectGeometry(
                silhouette_box_xywh=self.incoming_box_xywh,
                silhouette_confidence=1.0,
            )
        if not self.analysis_basis.strip():
            raise ValueError("boundary subject pair analysis basis cannot be blank")

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_basis": self.analysis_basis,
            "kind": self.kind,
            "outgoing_box_xywh": list(self.outgoing_box_xywh),
            "incoming_box_xywh": list(self.incoming_box_xywh),
        }


GeometryProvider = Callable[[np.ndarray], BoundaryFrameSubjectGeometry]


@dataclass(frozen=True, slots=True)
class BoundaryAffineDecision:
    boundary_index: int
    outgoing_source_index: int
    incoming_source_index: int
    outgoing_source_seconds: float
    incoming_source_seconds: float
    subject_kind: str | None
    result: BoundaryAffineMatchResult | None
    rejection_reason: str | None
    exact_subject_pair: BoundarySubjectPair | None = None

    def __post_init__(self) -> None:
        if self.boundary_index < 0:
            raise ValueError("boundary index cannot be negative")
        if min(self.outgoing_source_index, self.incoming_source_index) < 0:
            raise ValueError("source index cannot be negative")
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (
                self.outgoing_source_seconds,
                self.incoming_source_seconds,
            )
        ):
            raise ValueError("boundary source clocks must be finite and non-negative")
        if self.result is not None:
            if self.rejection_reason != self.result.rejection_reason:
                raise ValueError("decision and matcher rejection reasons disagree")
        elif not self.rejection_reason:
            raise ValueError("unmatched boundary decision needs a rejection reason")
        if self.exact_subject_pair is not None:
            if self.subject_kind != self.exact_subject_pair.kind:
                raise ValueError("decision and exact subject-pair kinds disagree")

    @property
    def eligible(self) -> bool:
        return bool(self.result is not None and self.result.eligible)

    @property
    def transform(self) -> BoundaryAffineTransformEvidence | None:
        return self.result.transform if self.result is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_index": self.boundary_index,
            "outgoing_source_index": self.outgoing_source_index,
            "incoming_source_index": self.incoming_source_index,
            "outgoing_source_seconds": self.outgoing_source_seconds,
            "incoming_source_seconds": self.incoming_source_seconds,
            "subject_kind": self.subject_kind,
            "exact_subject_pair": (
                self.exact_subject_pair.to_dict()
                if self.exact_subject_pair is not None
                else None
            ),
            "eligible": self.eligible,
            "rejection_reason": self.rejection_reason,
            "matcher": self.result.to_dict() if self.result is not None else None,
        }


@dataclass(frozen=True, slots=True)
class BoundaryAffinePlan:
    decisions: tuple[BoundaryAffineDecision, ...]
    analysis_basis: str = ANALYSIS_BASIS

    def __post_init__(self) -> None:
        indices = [item.boundary_index for item in self.decisions]
        if len(indices) != len(set(indices)):
            raise ValueError("boundary affine plan contains duplicate decisions")

    @property
    def eligible_count(self) -> int:
        return sum(item.eligible for item in self.decisions)

    @property
    def analyzed_count(self) -> int:
        return sum(item.result is not None for item in self.decisions)

    @property
    def exact_subject_pair_count(self) -> int:
        return sum(item.exact_subject_pair is not None for item in self.decisions)

    def decision_for(self, boundary_index: int) -> BoundaryAffineDecision | None:
        matches = [
            item for item in self.decisions if item.boundary_index == boundary_index
        ]
        if len(matches) > 1:
            raise ValueError("boundary affine plan contains duplicate decisions")
        return matches[0] if matches else None

    def to_dict(self) -> dict[str, Any]:
        rejection_counts: dict[str, int] = {}
        for item in self.decisions:
            if item.rejection_reason is not None:
                rejection_counts[item.rejection_reason] = (
                    rejection_counts.get(item.rejection_reason, 0) + 1
                )
        return {
            "analysis_basis": self.analysis_basis,
            "source_clock_basis": "selected_timeline_local_clock_last_output_frame",
            "frame_space": "854x480_center_cropped_output_space",
            "decision_count": len(self.decisions),
            "analyzed_count": self.analyzed_count,
            "eligible_count": self.eligible_count,
            "exact_subject_pair_count": self.exact_subject_pair_count,
            "rejection_counts": [
                {"reason": reason, "count": count}
                for reason, count in sorted(rejection_counts.items())
            ],
            "storage_policy": {
                "decoded_pixels_serialized": False,
                "feature_descriptors_serialized": False,
            },
            "decisions": [item.to_dict() for item in self.decisions],
        }


class _CaptureProvider:
    def __init__(self, sources: Sequence[Path], maximum_cache_size: int = 64) -> None:
        self._captures: list[cv2.VideoCapture | None] = []
        self._cache: OrderedDict[tuple[int, int], np.ndarray | None] = OrderedDict()
        self._maximum_cache_size = maximum_cache_size
        try:
            for path in sources:
                capture = cv2.VideoCapture(str(path))
                self._captures.append(capture if capture.isOpened() else None)
        except Exception:
            self.close()
            raise

    def frame_at(self, source_index: int, seconds: float) -> np.ndarray | None:
        key = (source_index, int(round(max(0.0, seconds) * 1000.0)))
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return None if value is None else value.copy()
        capture = self._captures[source_index]
        frame: np.ndarray | None = None
        if capture is not None:
            capture.set(cv2.CAP_PROP_POS_MSEC, key[1])
            success, decoded = capture.read()
            if success and isinstance(decoded, np.ndarray) and decoded.size:
                frame = decoded
        self._cache[key] = frame
        while len(self._cache) > self._maximum_cache_size:
            self._cache.popitem(last=False)
        return None if frame is None else frame.copy()

    def close(self) -> None:
        for capture in self._captures:
            if capture is not None:
                capture.release()
        self._captures.clear()
        self._cache.clear()


def _frame_counts(
    choices: Sequence[VisualTimelineChoice], fps: float
) -> tuple[int, ...]:
    counts: list[int] = []
    previous = 0
    cumulative = 0.0
    for choice in choices:
        cumulative += choice.duration_seconds
        boundary = int(round(cumulative * fps))
        count = boundary - previous
        if count < 2:
            raise ValueError("boundary affine segment is shorter than two frames")
        counts.append(count)
        previous = boundary
    return tuple(counts)


def _face_box(
    choice: VisualTimelineChoice, sample_index: int
) -> tuple[float, float, float, float] | None:
    box = choice.face_samples_xywh[sample_index]
    if box is None or choice.face_detector_backends[sample_index] == "none":
        return None
    if choice.face_match_candidate_failure is not None:
        return None
    return box


def _silhouette_box(
    choice: VisualTimelineChoice, sample_index: int
) -> tuple[float, float, float, float] | None:
    box = choice.silhouette_samples_xywh[sample_index]
    if box is None or choice.silhouette_backends[sample_index] == "none":
        return None
    if choice.silhouette_confidences[sample_index] < 0.72:
        return None
    return box


def _paired_subject(
    outgoing: VisualTimelineChoice,
    incoming: VisualTimelineChoice,
) -> tuple[
    str,
    tuple[float, float, float, float],
    tuple[float, float, float, float],
] | None:
    outgoing_face = _face_box(outgoing, 2)
    incoming_face = _face_box(incoming, 0)
    if outgoing_face is not None and incoming_face is not None:
        return ("face", outgoing_face, incoming_face)
    outgoing_silhouette = _silhouette_box(outgoing, 2)
    incoming_silhouette = _silhouette_box(incoming, 0)
    if outgoing_silhouette is not None and incoming_silhouette is not None:
        return ("silhouette", outgoing_silhouette, incoming_silhouette)
    return None


def _canonicalize(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, tuple[float, float, float, float] | None]:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("boundary frame must be a BGR image")
    source_height, source_width = frame.shape[:2]
    scale = max(width / source_width, height / source_height)
    scaled_width = max(width, int(math.ceil(source_width * scale)))
    scaled_height = max(height, int(math.ceil(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        frame,
        (scaled_width, scaled_height),
        interpolation=interpolation,
    )
    offset_x = (scaled_width - width) // 2
    offset_y = (scaled_height - height) // 2
    canonical = resized[offset_y : offset_y + height, offset_x : offset_x + width]
    x, y, box_width, box_height = box
    transformed = (
        (x * scaled_width - offset_x) / width,
        (y * scaled_height - offset_y) / height,
        box_width * scaled_width / width,
        box_height * scaled_height / height,
    )
    if (
        transformed[0] < -1e-6
        or transformed[1] < -1e-6
        or transformed[2] <= 0.0
        or transformed[3] <= 0.0
        or transformed[0] + transformed[2] > 1.000001
        or transformed[1] + transformed[3] > 1.000001
    ):
        return (canonical, None)
    return (
        canonical,
        tuple(min(1.0, max(0.0, value)) for value in transformed),
    )


def _canonical_frame(
    frame: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Center-crop one decoded frame exactly as the full-frame renderer does."""

    # The box is merely a guaranteed in-bounds marker.  Keeping this wrapper
    # beside ``_canonicalize`` prevents the pixel and legacy-box paths from
    # drifting to different resize/crop arithmetic.
    canonical, _box = _canonicalize(
        frame,
        (0.49, 0.49, 0.02, 0.02),
        width=width,
        height=height,
    )
    return canonical


def _default_geometry_provider(frame: np.ndarray) -> BoundaryFrameSubjectGeometry:
    # Load the existing bundled detectors lazily.  This runs only for the two
    # endpoint frames of a real source switch, not for the whole performance.
    from .visual_score import describe_frame

    descriptor = describe_frame(
        frame,
        include_subject_geometry=True,
        include_formation_pose=False,
    )
    return BoundaryFrameSubjectGeometry(
        face_count=descriptor.face_count,
        face_box_xywh=descriptor.primary_face_bbox,
        silhouette_box_xywh=descriptor.silhouette_bbox_xywh,
        silhouette_confidence=descriptor.silhouette_confidence,
    )


def _exact_subject_pair(
    outgoing: BoundaryFrameSubjectGeometry,
    incoming: BoundaryFrameSubjectGeometry,
    *,
    minimum_silhouette_confidence: float = 0.72,
) -> BoundarySubjectPair | None:
    # A dominant face in a group shot can be a different member on each side.
    # Face-driven matching is therefore limited to an unambiguous solo face;
    # group frames use the complete performance silhouette instead.
    if (
        outgoing.face_count == 1
        and incoming.face_count == 1
        and outgoing.face_box_xywh is not None
        and incoming.face_box_xywh is not None
    ):
        return BoundarySubjectPair(
            "face",
            outgoing.face_box_xywh,
            incoming.face_box_xywh,
        )
    if (
        outgoing.silhouette_box_xywh is not None
        and incoming.silhouette_box_xywh is not None
        and outgoing.silhouette_confidence >= minimum_silhouette_confidence
        and incoming.silhouette_confidence >= minimum_silhouette_confidence
    ):
        return BoundarySubjectPair(
            "silhouette",
            outgoing.silhouette_box_xywh,
            incoming.silhouette_box_xywh,
        )
    return None


def _face_match_segments(face_match_plan: FaceMatchPlan | None) -> dict[int, bool]:
    if face_match_plan is None:
        return {}
    output = {item.segment_index: item.applied for item in face_match_plan.segments}
    if len(output) != len(face_match_plan.segments):
        raise ValueError("face-match plan contains duplicate segment indices")
    return output


def build_boundary_affine_plan(
    *,
    visual_timeline: VisualTimelinePlan,
    sources: Sequence[Path],
    transitions: Sequence[RenderTransition],
    face_match_plan: FaceMatchPlan | None = None,
    fps: float = 29.97,
    output_width: int = 854,
    output_height: int = 480,
    frame_provider: FrameProvider | None = None,
    geometry_provider: GeometryProvider | None = None,
) -> BoundaryAffinePlan:
    """Review every real hard-cut source switch using actual decoded frames."""

    choices = visual_timeline.choices
    if not choices:
        raise ValueError("visual timeline cannot be empty")
    if len(transitions) != len(choices) - 1:
        raise ValueError("boundary affine planner needs one transition per boundary")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("boundary affine fps must be positive and finite")
    if output_width < 160 or output_height < 90:
        raise ValueError("boundary affine output size is too small")
    if any(not 0 <= item.source_index < len(sources) for item in choices):
        raise ValueError("visual timeline references an unknown source")

    counts = _frame_counts(choices, fps)
    base_crops = _face_match_segments(face_match_plan)
    capture_provider: _CaptureProvider | None = None
    if frame_provider is None:
        capture_provider = _CaptureProvider(tuple(Path(item) for item in sources))
        provider = capture_provider.frame_at
    else:
        provider = frame_provider
    subject_provider = geometry_provider or _default_geometry_provider
    decisions: list[BoundaryAffineDecision] = []

    def rejected(
        boundary_index: int,
        left: VisualTimelineChoice,
        right: VisualTimelineChoice,
        outgoing_seconds: float,
        reason: str,
        subject_kind: str | None = None,
        exact_subject_pair: BoundarySubjectPair | None = None,
    ) -> None:
        decisions.append(
            BoundaryAffineDecision(
                boundary_index,
                left.source_index,
                right.source_index,
                outgoing_seconds,
                right.source_local_start,
                subject_kind,
                None,
                reason,
                exact_subject_pair,
            )
        )

    try:
        for boundary_index, (left, right) in enumerate(zip(choices, choices[1:])):
            if left.source_index == right.source_index:
                continue
            playback_rate = left.source_playback_rate
            outgoing_seconds = min(
                left.source_local_end,
                left.source_local_start
                + (counts[boundary_index] - 1) * playback_rate / fps,
            )
            transition = transitions[boundary_index]
            if transition.kind is not TransitionKind.HARD_CUT:
                rejected(
                    boundary_index,
                    left,
                    right,
                    outgoing_seconds,
                    "non_hard_cut_transition",
                )
                continue
            if left.internal_scene_cut_master_seconds or right.internal_scene_cut_master_seconds:
                rejected(
                    boundary_index,
                    left,
                    right,
                    outgoing_seconds,
                    "internal_camera_cut",
                )
                continue
            if base_crops.get(left.segment_index) or base_crops.get(right.segment_index):
                rejected(
                    boundary_index,
                    left,
                    right,
                    outgoing_seconds,
                    "base_pan_crop_coordinate_space_mismatch",
                )
                continue
            outgoing_frame = provider(left.source_index, outgoing_seconds)
            incoming_frame = provider(right.source_index, right.source_local_start)
            if outgoing_frame is None or incoming_frame is None:
                rejected(
                    boundary_index,
                    left,
                    right,
                    outgoing_seconds,
                    "boundary_frame_decode_failed",
                )
                continue
            canonical_outgoing = _canonical_frame(
                outgoing_frame,
                width=output_width,
                height=output_height,
            )
            canonical_incoming = _canonical_frame(
                incoming_frame,
                width=output_width,
                height=output_height,
            )
            exact_pair: BoundarySubjectPair | None = None
            try:
                exact_pair = _exact_subject_pair(
                    subject_provider(canonical_outgoing),
                    subject_provider(canonical_incoming),
                )
            except Exception:
                # Geometry is an optional bounded correction layer.  A local
                # detector failure must not abort candidate materialization;
                # the legacy sampled geometry remains a fail-closed fallback.
                exact_pair = None
            if exact_pair is not None:
                subject_kind = exact_pair.kind
                canonical_outgoing_box = exact_pair.outgoing_box_xywh
                canonical_incoming_box = exact_pair.incoming_box_xywh
            else:
                paired = _paired_subject(left, right)
                if paired is None:
                    rejected(
                        boundary_index,
                        left,
                        right,
                        outgoing_seconds,
                        "common_subject_geometry_unavailable",
                    )
                    continue
                subject_kind, outgoing_box, incoming_box = paired
                _unused_outgoing, canonical_outgoing_box = _canonicalize(
                    outgoing_frame,
                    outgoing_box,
                    width=output_width,
                    height=output_height,
                )
                _unused_incoming, canonical_incoming_box = _canonicalize(
                    incoming_frame,
                    incoming_box,
                    width=output_width,
                    height=output_height,
                )
                if (
                    canonical_outgoing_box is None
                    or canonical_incoming_box is None
                ):
                    rejected(
                        boundary_index,
                        left,
                        right,
                        outgoing_seconds,
                        "subject_clipped_by_output_canonicalization",
                        subject_kind,
                    )
                    continue
            result = match_boundary_affine(
                canonical_outgoing,
                canonical_incoming,
                outgoing_subject_box=canonical_outgoing_box,
                incoming_subject_box=canonical_incoming_box,
            )
            decisions.append(
                BoundaryAffineDecision(
                    boundary_index,
                    left.source_index,
                    right.source_index,
                    outgoing_seconds,
                    right.source_local_start,
                    subject_kind,
                    result,
                    result.rejection_reason,
                    exact_pair,
                )
            )
    finally:
        if capture_provider is not None:
            capture_provider.close()
    return BoundaryAffinePlan(tuple(decisions))


__all__ = [
    "ANALYSIS_BASIS",
    "BoundaryAffineDecision",
    "BoundaryAffinePlan",
    "BoundaryFrameSubjectGeometry",
    "BoundarySubjectPair",
    "FrameProvider",
    "GeometryProvider",
    "build_boundary_affine_plan",
]
