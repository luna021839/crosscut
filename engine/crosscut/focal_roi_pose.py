"""Stateless focal-body pose inference, isolated from production descriptors.

A stable detector box can still produce a confidently wrong full-frame pose
when a neighbor enters MediaPipe's virtual-body ROI. This analyzer limits the
analysis pixels to one face-associated body before proposing a pose. It changes
neither output framing nor any cut/identity policy. Callers must independently
validate temporal continuity and fingerprint this implementation before caching
its results. The eight 2D joints are not a whole-body pose embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .focal_pose import face_associated_person
from .pose import FormationPersonDescriptor, SubjectGeometryAnalyzer, _upper_body_shape


_JOINTS = (11, 12, 13, 14, 15, 16, 23, 24)
_TORSO_JOINTS = (11, 12, 23, 24)


@dataclass(frozen=True, slots=True)
class FocalRoiPoseResult:
    reason: str
    person: FormationPersonDescriptor | None = None
    roi_xyxy: tuple[int, int, int, int] | None = None
    screen_upper_body_xyc: tuple[tuple[float, float, float], ...] = ()
    proposal_count: int = 0
    associated_observation_count: int = 0


class FocalRoiPoseAnalyzer:
    """Own lazy model state; no temporal smoothing or history-dependent results."""

    def __init__(self, model_directory: Path | None = None) -> None:
        self._geometry = SubjectGeometryAnalyzer(model_directory)

    def describe(
        self,
        image: np.ndarray,
        *,
        face_bbox_xywh: tuple[float, float, float, float] | None,
        people: Sequence[FormationPersonDescriptor],
    ) -> FocalRoiPoseResult:
        if (
            not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] != 3 or not image.size or image.dtype != np.uint8
        ):
            raise ValueError("focal pose expects a non-empty uint8 BGR image")
        person = face_associated_person(face_bbox_xywh, people)
        if person is None:
            return FocalRoiPoseResult("no_unique_face_body")
        height, width = image.shape[:2]
        bx, by, bw, bh = person.bbox_xywh
        # Preserve hands near the box edge without returning to the original
        # virtual-body radius that admitted the neighboring performer's torso.
        roi_box = (
            max(0, int(np.floor((bx - .1 * bw) * width))),
            max(0, int(np.floor((by - .05 * bh) * height))),
            min(width, int(np.ceil((bx + 1.1 * bw) * width))),
            min(height, int(np.ceil((by + 1.05 * bh) * height))),
        )
        x0, y0, x1, y1 = roi_box
        if x1 - x0 < 8 or y1 - y0 < 8:
            return FocalRoiPoseResult("insufficient_roi_pixels", roi_xyxy=roi_box)
        roi = image[y0:y1, x0:x1]
        assert face_bbox_xywh is not None  # established by geometric association
        fx, fy, fw, fh = face_bbox_xywh
        face_center = np.asarray(((fx + fw / 2) * width - x0,
                                  (fy + fh / 2) * height - y0))
        face_size = np.asarray((fw * width, fh * height))
        selected = []
        with self._geometry._lock:
            self._geometry._load_pose()
            detector = self._geometry._pose_detector
            estimator = self._geometry._pose_estimator
            if detector is None or estimator is None:
                return FocalRoiPoseResult("pose_model_unavailable", roi_xyxy=roi_box)
            try:
                proposals = detector.infer(roi)
                for proposal in proposals:
                    observation = estimator.infer(roi, proposal)
                    if observation is None:
                        continue
                    screen = observation.screen_landmarks
                    if (
                        not isinstance(screen, np.ndarray) or screen.shape != (33, 5)
                        or not np.isfinite(screen).all()
                        or not np.isfinite(observation.confidence)
                        or not .5 <= observation.confidence <= 1
                    ):
                        continue
                    reliability = np.sqrt(np.clip(screen[:, 3], 0, 1)
                                          * np.clip(screen[:, 4], 0, 1))
                    if reliability[0] < .35:
                        continue
                    distance = float(np.linalg.norm((screen[0, :2] - face_center) / face_size))
                    if distance > .8:
                        continue
                    # Ambiguity precedes the quality check: a weak focal pose
                    # must not cause another face-associated observation to win.
                    points = _upper_body_shape(observation)
                    torso = screen[list(_TORSO_JOINTS), :2]
                    torso_inside = bool(
                        np.all(torso >= 0)
                        and np.all(torso[:, 0] < roi.shape[1])
                        and np.all(torso[:, 1] < roi.shape[0])
                    )
                    selected.append((observation, points if torso_inside else ()))
            except (cv2.error, ValueError, FloatingPointError):
                return FocalRoiPoseResult("pose_inference_failed", roi_xyxy=roi_box)
        counts = {"roi_xyxy": roi_box, "proposal_count": len(proposals),
                  "associated_observation_count": len(selected)}
        if len(selected) != 1:
            return FocalRoiPoseResult(
                "ambiguous_pose" if len(selected) > 1 else "no_face_associated_pose", **counts
            )
        observation, points = selected[0]
        if not points:
            return FocalRoiPoseResult("unreliable_upper_body", **counts)
        screen = observation.screen_landmarks
        visible = np.sqrt(np.clip(screen[:, 3], 0, 1) * np.clip(screen[:, 4], 0, 1))
        screen_points = tuple(
            (float((screen[index, 0] + x0) / width),
             float((screen[index, 1] + y0) / height),
             float(visible[index] * observation.confidence))
            for index in _JOINTS
        )
        return FocalRoiPoseResult(
            "observed", person=replace(person, upper_body_keypoints_xyc=points),
            screen_upper_body_xyc=screen_points, **counts
        )
