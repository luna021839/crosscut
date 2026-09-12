"""Transient focal-face correspondence for verified member fancams.

Automatic member acquisition already proves that every input is an official
fancam for the requested member.  A frame can still contain several IVE
members, so treating an aggregate silhouette as the focal performer produces
the wrong crop.  This module does not infer a name.  It matches YuNet faces
across the exact cut window with local SFace appearance evidence and a modest
fancam-center prior.  Embeddings remain in memory and are never serialized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import cv2
import numpy as np

from .face_identity import cosine_similarity, extract_sface_embedding
from .visual_score import (
    _detect_faces,
    _normalized_face_metadata,
    _working_frame,
)


FOCAL_FACE_BASIS = "verified_official_fancam_focal_face_correspondence"


@dataclass(frozen=True, slots=True)
class FocalFaceCandidate:
    bbox_xywh: tuple[float, float, float, float]
    embedding: tuple[float, ...] = field(repr=False)
    eye_scale: float | None = None
    landmark_scale: float | None = None

    def __post_init__(self) -> None:
        if len(self.bbox_xywh) != 4 or not self.embedding:
            raise ValueError("focal face candidate evidence is incomplete")
        x, y, width, height = (float(value) for value in self.bbox_xywh)
        if (
            not all(math.isfinite(value) for value in (x, y, width, height))
            or x < 0.0
            or y < 0.0
            or width <= 0.0
            or height <= 0.0
            or x + width > 1.000001
            or y + height > 1.000001
            or any(not math.isfinite(float(value)) for value in self.embedding)
        ):
            raise ValueError("focal face candidate values are invalid")
        if self.eye_scale is not None and (
            not math.isfinite(float(self.eye_scale))
            or float(self.eye_scale) <= 0.0
        ):
            raise ValueError("focal face eye scale is invalid")
        if self.landmark_scale is not None and (
            not math.isfinite(float(self.landmark_scale))
            or float(self.landmark_scale) <= 0.0
        ):
            raise ValueError("focal face landmark scale is invalid")

    @property
    def center_xy(self) -> tuple[float, float]:
        x, y, width, height = self.bbox_xywh
        return (x + width / 2.0, y + height / 2.0)


@dataclass(frozen=True, slots=True)
class FocalFacePair:
    outgoing: FocalFaceCandidate
    incoming: FocalFaceCandidate
    cosine_similarity: float
    selection_score: float


def _landmark_rms_scale(
    landmarks_xy: Sequence[tuple[float, float]],
    *,
    frame_width: int,
    frame_height: int,
) -> float | None:
    if not landmarks_xy or frame_width <= 0 or frame_height <= 0:
        return None
    aspect = frame_width / frame_height
    points = tuple(
        (float(x) * aspect, float(y)) for x, y in landmarks_xy
    )
    if any(
        not math.isfinite(value)
        for point in points
        for value in point
    ):
        return None
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    scale = math.sqrt(
        sum(
            (point[0] - center_x) ** 2 + (point[1] - center_y) ** 2
            for point in points
        )
        / len(points)
    )
    return scale if scale > 0.0 and math.isfinite(scale) else None


def detect_focal_face_candidates(
    frame: np.ndarray,
) -> tuple[FocalFaceCandidate, ...]:
    """Return every locally embeddable YuNet face in one decoded frame."""

    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("focal face detection needs a decoded frame")
    working = _working_frame(frame)
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    faces, backend = _detect_faces(working, gray)
    if backend.casefold() != "yunet":
        return ()
    candidates: list[FocalFaceCandidate] = []
    for face in faces:
        (
            bbox,
            landmarks,
            _roll,
            eye_scale,
            _geometry,
        ) = _normalized_face_metadata(
            face,
            frame_width=working.shape[1],
            frame_height=working.shape[0],
        )
        if bbox is None or landmarks is None:
            continue
        embedding = extract_sface_embedding(
            working,
            bbox_xywh=bbox,
            landmarks_xy=landmarks,
        )
        if embedding:
            candidates.append(
                FocalFaceCandidate(
                    bbox,
                    embedding,
                    eye_scale=(
                        float(eye_scale)
                        if eye_scale is not None
                        else None
                    ),
                    landmark_scale=_landmark_rms_scale(
                        landmarks,
                        frame_width=working.shape[1],
                        frame_height=working.shape[0],
                    ),
                )
            )
    return tuple(candidates)


def _focal_prior(candidate: FocalFaceCandidate) -> float:
    center_x, center_y = candidate.center_xy
    normalized_distance = math.hypot(
        (center_x - 0.50) / 0.50,
        (center_y - 0.40) / 0.60,
    )
    center_score = max(0.0, 1.0 - normalized_distance)
    area = candidate.bbox_xywh[2] * candidate.bbox_xywh[3]
    scale_score = min(1.0, math.sqrt(area) / 0.18)
    return 0.82 * center_score + 0.18 * scale_score


def select_focal_face_pair(
    outgoing: Sequence[FocalFaceCandidate],
    incoming: Sequence[FocalFaceCandidate],
    *,
    minimum_similarity: float = 0.36,
    minimum_score_margin: float = 0.025,
) -> FocalFacePair | None:
    """Select one same-face cut pair without assigning a named identity."""

    if not 0.0 <= minimum_similarity <= 1.0:
        raise ValueError("focal face similarity threshold is invalid")
    if minimum_score_margin < 0.0 or not math.isfinite(minimum_score_margin):
        raise ValueError("focal face ambiguity margin is invalid")
    ranked: list[FocalFacePair] = []
    for left in outgoing:
        for right in incoming:
            similarity = cosine_similarity(left.embedding, right.embedding)
            if similarity is None or similarity + 1e-9 < minimum_similarity:
                continue
            height_ratio = max(
                left.bbox_xywh[3],
                right.bbox_xywh[3],
            ) / max(
                1e-9,
                min(left.bbox_xywh[3], right.bbox_xywh[3]),
            )
            score = (
                similarity
                + 0.16 * min(_focal_prior(left), _focal_prior(right))
                + 0.06 * (
                    _focal_prior(left) + _focal_prior(right)
                ) / 2.0
                - 0.025 * abs(math.log(height_ratio))
            )
            ranked.append(FocalFacePair(left, right, similarity, score))
    if not ranked:
        return None
    ranked.sort(
        key=lambda item: (
            item.selection_score,
            item.cosine_similarity,
            _focal_prior(item.outgoing) + _focal_prior(item.incoming),
        ),
        reverse=True,
    )
    best = ranked[0]
    if len(ranked) > 1:
        second = ranked[1]
        independent_alternative = bool(
            second.outgoing is not best.outgoing
            and second.incoming is not best.incoming
        )
        if (
            independent_alternative
            and best.selection_score - second.selection_score
            < minimum_score_margin - 1e-12
        ):
            return None
    return best


def select_focal_face_matching_reference(
    candidates: Sequence[FocalFaceCandidate],
    references: Sequence[FocalFaceCandidate],
    *,
    minimum_similarity: float = 0.36,
) -> FocalFaceCandidate | None:
    """Extend a verified endpoint correspondence by one adjacent frame."""

    ranked: list[tuple[float, float, FocalFaceCandidate]] = []
    for candidate in candidates:
        similarities = tuple(
            value
            for reference in references
            if (
                value := cosine_similarity(
                    candidate.embedding,
                    reference.embedding,
                )
            )
            is not None
        )
        if not similarities:
            continue
        similarity = max(similarities)
        if similarity + 1e-9 < minimum_similarity:
            continue
        ranked.append(
            (
                similarity + 0.08 * _focal_prior(candidate),
                similarity,
                candidate,
            )
        )
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.02 - 1e-12:
        return None
    return ranked[0][2]


__all__ = [
    "FOCAL_FACE_BASIS",
    "FocalFaceCandidate",
    "FocalFacePair",
    "detect_focal_face_candidates",
    "select_focal_face_matching_reference",
    "select_focal_face_pair",
]
