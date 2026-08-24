"""Conservative first-anchor inference for member-specific IVE fancams.

Acquisition metadata confirms which member the fancam claims to feature. This
module does not perform face recognition or re-identification: it only accepts
an early frame where one clearly dominant, centered person remains
geometrically stable across adjacent samples. Smaller or clearly subordinate
background dancers may remain visible; locally comparable candidates are
rejected before tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from .pose import PersonDetection, detect_people
from .request import MemberAnchorRequest


FrameReader = Callable[[Path, float], np.ndarray | None]
PersonDetector = Callable[[np.ndarray], Sequence[PersonDetection]]


@dataclass(frozen=True, slots=True)
class AutoAnchorResult:
    source: Path
    accepted: bool
    anchor: MemberAnchorRequest | None
    confidence: float
    sampled_times_seconds: tuple[float, ...]
    rejection_reasons: tuple[str, ...]
    policy: str = "centered_dominant_subject_temporal_stability_v3"

    def to_dict(self) -> dict[str, object]:
        return {
            "source": str(self.source),
            "accepted": self.accepted,
            "anchor": self.anchor.as_dict() if self.anchor else None,
            "confidence": self.confidence,
            "sampled_times_seconds": list(self.sampled_times_seconds),
            "rejection_reasons": list(self.rejection_reasons),
            "policy": self.policy,
            "identity_claim": "not_established_no_face_reference_geometry_only",
        }


_STABILITY_CHECKPOINT_OFFSETS_SECONDS = (
    -0.75,
    -0.50,
    -0.25,
    0.0,
    0.25,
    0.50,
    0.75,
)
_STABILITY_HALF_WINDOW_SECONDS = max(
    abs(offset) for offset in _STABILITY_CHECKPOINT_OFFSETS_SECONDS
)
_MINIMUM_ADJACENT_IOU = 0.24
_MAXIMUM_ADJACENT_CENTER_DISTANCE = 0.22
_MINIMUM_ANCHOR_CONFIDENCE = 0.50


def _read_frame(source: Path, time_seconds: float) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_seconds) * 1000.0)
        success, frame = capture.read()
        if not success or frame is None or frame.size == 0:
            return None
        return frame
    finally:
        capture.release()


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection_left = max(left[0], right[0])
    intersection_top = max(left[1], right[1])
    intersection_right = min(left[2], right[2])
    intersection_bottom = min(left[3], right[3])
    intersection = max(0.0, intersection_right - intersection_left) * max(
        0.0, intersection_bottom - intersection_top
    )
    union = _box_area(left) + _box_area(right) - intersection
    return intersection / union if union > 1e-9 else 0.0


def _candidate_score(candidate: PersonDetection) -> float:
    box = candidate.bbox_xyxy
    center_x, center_y = _box_center(box)
    center_distance = math.hypot(center_x - 0.5, center_y - 0.50) / 0.72
    center_score = max(0.0, 1.0 - center_distance)
    area_score = min(1.0, math.sqrt(_box_area(box) / 0.18))
    return 0.48 * candidate.confidence + 0.34 * center_score + 0.18 * area_score


def _dominant_candidate(
    detections: Sequence[PersonDetection],
) -> tuple[PersonDetection | None, str | None]:
    usable: list[PersonDetection] = []
    for detection in detections:
        left, top, right, bottom = detection.bbox_xyxy
        width = right - left
        height = bottom - top
        center_x, center_y = _box_center(detection.bbox_xyxy)
        if (
            detection.confidence < 0.48
            or width < 0.06
            or height < 0.18
            or _box_area(detection.bbox_xyxy) < 0.018
            or abs(center_x - 0.5) > 0.38
            or abs(center_y - 0.52) > 0.42
        ):
            continue
        usable.append(detection)
    if not usable:
        return (None, "no_safe_center_person")
    ranked = sorted(
        usable,
        key=lambda item: (-_candidate_score(item), -item.confidence, item.bbox_xyxy),
    )
    if len(ranked) > 1:
        first_score = _candidate_score(ranked[0])
        second_score = _candidate_score(ranked[1])
        if second_score >= first_score - 0.065:
            return (None, "ambiguous_comparable_people")
    return (ranked[0], None)


def _expand_box(
    box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    return (
        max(0.0, left - width * 0.04),
        max(0.0, top - height * 0.025),
        min(1.0, right + width * 0.04),
        min(1.0, bottom + height * 0.025),
    )


def infer_member_anchor(
    source: Path,
    *,
    source_index: int,
    duration_seconds: float,
    canonical_start_seconds: float,
    audio_offset_seconds: float,
    scene_change_seconds: Sequence[float] = (),
    frame_reader: FrameReader = _read_frame,
    person_detector: PersonDetector = detect_people,
) -> AutoAnchorResult:
    """Infer one stable start box, or return an explicit rejection."""

    if duration_seconds <= 0 or not math.isfinite(duration_seconds):
        raise ValueError("duration_seconds must be positive and finite")
    base_local_time = max(0.0, canonical_start_seconds + audio_offset_seconds)
    offsets = (0.8, 1.4, 2.0, 2.8, 3.8, 5.0, 6.5, 8.0)
    candidate_times = tuple(
        base_local_time + offset
        for offset in offsets
        if base_local_time + offset - _STABILITY_HALF_WINDOW_SECONDS >= 0.0
        and base_local_time + offset + _STABILITY_HALF_WINDOW_SECONDS
        < duration_seconds
    )
    sampled: list[float] = []
    reasons: list[str] = []
    for center_time in candidate_times:
        window_start = center_time - _STABILITY_HALF_WINDOW_SECONDS
        window_end = center_time + _STABILITY_HALF_WINDOW_SECONDS
        if any(
            window_start - 0.12 <= cut <= window_end + 0.12
            for cut in scene_change_seconds
        ):
            reasons.append("near_scene_change")
            continue
        times = tuple(
            center_time + offset
            for offset in _STABILITY_CHECKPOINT_OFFSETS_SECONDS
        )
        selected: list[PersonDetection | None] = []
        local_reasons: list[str] = []
        for sample_time in times:
            sampled.append(sample_time)
            frame = frame_reader(source, sample_time)
            if frame is None:
                selected.append(None)
                local_reasons.append("frame_read_failed")
                continue
            detection, reason = _dominant_candidate(person_detector(frame))
            selected.append(detection)
            if reason:
                local_reasons.append(reason)
        if any(item is None for item in selected):
            reasons.extend(local_reasons)
            reasons.append("single_subject_not_continuous_across_window")
            continue
        stable = tuple(item for item in selected if item is not None)
        adjacent_ious: list[float] = []
        adjacent_center_distances: list[float] = []
        for previous, current in zip(stable, stable[1:]):
            adjacent_ious.append(_iou(previous.bbox_xyxy, current.bbox_xyxy))
            adjacent_center_distances.append(
                math.dist(
                    _box_center(previous.bbox_xyxy),
                    _box_center(current.bbox_xyxy),
                )
            )
        if any(iou < _MINIMUM_ADJACENT_IOU for iou in adjacent_ious) or any(
            distance > _MAXIMUM_ADJACENT_CENTER_DISTANCE
            for distance in adjacent_center_distances
        ):
            reasons.append("single_subject_not_continuous_across_window")
            continue
        center = stable[len(stable) // 2]
        # One soft detector frame is common during fast choreography.  The
        # full seven-checkpoint IoU path is stronger evidence than making the
        # weakest single confidence dominate the whole window, so use the
        # second-lowest observation while still requiring every frame >= 0.48.
        robust_detection_confidence = sorted(
            item.confidence for item in stable
        )[1]
        mean_adjacent_iou = sum(adjacent_ious) / len(adjacent_ious)
        maximum_center_distance = max(adjacent_center_distances)
        motion_stability = max(
            0.0,
            1.0
            - maximum_center_distance / _MAXIMUM_ADJACENT_CENTER_DISTANCE,
        )
        confidence = min(
            1.0,
            0.55 * robust_detection_confidence
            + 0.30 * mean_adjacent_iou
            + 0.15 * motion_stability,
        )
        if confidence < _MINIMUM_ANCHOR_CONFIDENCE:
            reasons.append("anchor_confidence_too_low")
            continue
        anchor = MemberAnchorRequest(
            source_index=source_index,
            time_seconds=center_time,
            bbox_xyxy=_expand_box(center.bbox_xyxy),
        )
        return AutoAnchorResult(
            source=source,
            accepted=True,
            anchor=anchor,
            confidence=confidence,
            sampled_times_seconds=tuple(sampled),
            rejection_reasons=tuple(dict.fromkeys(reasons)),
        )
    return AutoAnchorResult(
        source=source,
        accepted=False,
        anchor=None,
        confidence=0.0,
        sampled_times_seconds=tuple(sampled),
        rejection_reasons=tuple(dict.fromkeys(reasons or ["no_safe_early_sample"])),
    )


__all__ = ["AutoAnchorResult", "infer_member_anchor"]
