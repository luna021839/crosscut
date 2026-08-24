from __future__ import annotations

import math
from pathlib import Path
import unittest

import cv2
import numpy as np

from crosscut.boundary_affine_plan import (
    BoundaryFrameSubjectGeometry,
    build_boundary_affine_plan,
)
from crosscut.face_match import FaceMatchPlan, FaceMatchSegment
from crosscut.render import (
    NormalizedCrop,
    PanCropKeyframe,
    PanCropTrack,
    RenderTransition,
)
from crosscut.timeline import ShotType
from crosscut.visual_score import VisualTimelineChoice, VisualTimelinePlan


def _rich_frame(width: int = 640, height: int = 360) -> np.ndarray:
    random = np.random.default_rng(8842)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for index in range(180):
        x = int(random.integers(18, width - 18))
        y = int(random.integers(18, height - 18))
        color = tuple(int(value) for value in random.integers(30, 255, 3))
        cv2.circle(frame, (x, y), 3 + index % 7, color, -1)
    return frame


def _known_pair() -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    outgoing = _rich_frame()
    height, width = outgoing.shape[:2]
    scale = 1.09
    radians = math.radians(1.0)
    linear = np.asarray(
        [
            [scale * math.cos(radians), -scale * math.sin(radians)],
            [scale * math.sin(radians), scale * math.cos(radians)],
        ],
        dtype=np.float64,
    )
    target = np.asarray([0.492 * width, 0.506 * height])
    translation = target - linear @ np.asarray([width / 2.0, height / 2.0])
    incoming_to_outgoing = np.column_stack((linear, translation))
    incoming = cv2.warpAffine(
        outgoing,
        cv2.invertAffineTransform(incoming_to_outgoing),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    incoming_box = (0.35, 0.25, 0.18, 0.30)
    corners = np.asarray(
        [[
            [incoming_box[0] * width, incoming_box[1] * height],
            [(incoming_box[0] + incoming_box[2]) * width, incoming_box[1] * height],
            [incoming_box[0] * width, (incoming_box[1] + incoming_box[3]) * height],
            [
                (incoming_box[0] + incoming_box[2]) * width,
                (incoming_box[1] + incoming_box[3]) * height,
            ],
        ]],
        dtype=np.float32,
    )
    transformed = cv2.transform(corners, incoming_to_outgoing)[0]
    left, top = transformed.min(axis=0)
    right, bottom = transformed.max(axis=0)
    outgoing_box = (
        float(left / width),
        float(top / height),
        float((right - left) / width),
        float((bottom - top) / height),
    )
    return outgoing, incoming, outgoing_box, incoming_box


def _choice(
    segment_index: int,
    source_index: int,
    *,
    master_start: float,
    master_end: float,
    source_start: float,
    source_end: float,
    face_box: tuple[float, float, float, float],
) -> VisualTimelineChoice:
    return VisualTimelineChoice(
        segment_index=segment_index,
        source_index=source_index,
        canonical_master_start=master_start,
        canonical_master_end=master_end,
        source_local_start=source_start,
        source_local_end=source_end,
        quality_score=0.9,
        composition_score=0.9,
        boundary_score=0.9,
        shot_scale=0.4,
        shot_type=ShotType.MEDIUM,
        shot_id=f"{source_index}:shot",
        face_samples_xywh=(face_box, face_box, face_box),
        face_detector_backends=("opencv_zoo_yunet_2023mar",) * 3,
    )


def _timeline(
    outgoing_box: tuple[float, float, float, float],
    incoming_box: tuple[float, float, float, float],
) -> VisualTimelinePlan:
    return VisualTimelinePlan(
        (
            _choice(
                0,
                0,
                master_start=0.0,
                master_end=1.0,
                source_start=3.0,
                source_end=4.0,
                face_box=outgoing_box,
            ),
            _choice(
                1,
                1,
                master_start=1.0,
                master_end=2.0,
                source_start=8.0,
                source_end=9.0,
                face_box=incoming_box,
            ),
        ),
        1.0,
    )


class BoundaryAffinePlanTests(unittest.TestCase):
    def test_exact_boundary_clocks_and_real_frames_produce_eligible_plan(self) -> None:
        outgoing, incoming, outgoing_box, incoming_box = _known_pair()
        calls: list[tuple[int, float]] = []

        def provider(source_index: int, seconds: float) -> np.ndarray:
            calls.append((source_index, seconds))
            return outgoing.copy() if source_index == 0 else incoming.copy()

        exact_geometries = iter(
            (
                BoundaryFrameSubjectGeometry(
                    face_count=1,
                    face_box_xywh=outgoing_box,
                ),
                BoundaryFrameSubjectGeometry(
                    face_count=1,
                    face_box_xywh=incoming_box,
                ),
            )
        )

        plan = build_boundary_affine_plan(
            visual_timeline=_timeline(outgoing_box, incoming_box),
            sources=(Path("left.mp4"), Path("right.mp4")),
            transitions=(RenderTransition(tier="C", reason="hard"),),
            fps=30.0,
            output_width=640,
            output_height=360,
            frame_provider=provider,
            geometry_provider=lambda _frame: next(exact_geometries),
        )

        self.assertEqual(plan.analyzed_count, 1)
        self.assertEqual(plan.eligible_count, 1)
        decision = plan.decisions[0]
        self.assertAlmostEqual(decision.outgoing_source_seconds, 3.0 + 29 / 30)
        self.assertEqual(decision.incoming_source_seconds, 8.0)
        self.assertEqual(calls, [(0, 3.0 + 29 / 30), (1, 8.0)])
        self.assertEqual(decision.subject_kind, "face")
        self.assertIsNotNone(decision.exact_subject_pair)
        self.assertEqual(plan.exact_subject_pair_count, 1)
        self.assertIsNotNone(decision.transform)
        payload = plan.to_dict()
        self.assertFalse(payload["storage_policy"]["decoded_pixels_serialized"])
        self.assertFalse(
            payload["decisions"][0]["matcher"]["storage_policy"][
                "pixels_serialized"
            ]
        )

    def test_existing_base_crop_rejects_raw_coordinate_affine_without_decoding(self) -> None:
        _outgoing, _incoming, outgoing_box, incoming_box = _known_pair()
        crop = PanCropTrack(
            (PanCropKeyframe(0.0, NormalizedCrop(0.1, 0.1, 0.8, 0.8)),)
        )
        face_match = FaceMatchPlan(
            (
                FaceMatchSegment(
                    0,
                    0,
                    True,
                    None,
                    0.3,
                    0.3,
                    1.25,
                    1.25,
                    crop,
                    "face",
                ),
                FaceMatchSegment(
                    1,
                    1,
                    False,
                    "raw",
                    0.3,
                    0.3,
                    1.0,
                    None,
                    None,
                ),
            ),
            0.3,
            1.35,
            (0.5, 0.42),
        )

        def provider(_source_index: int, _seconds: float) -> np.ndarray:
            self.fail("base-crop rejection must happen before frame decoding")

        plan = build_boundary_affine_plan(
            visual_timeline=_timeline(outgoing_box, incoming_box),
            sources=(Path("left.mp4"), Path("right.mp4")),
            transitions=(RenderTransition(tier="B", reason="hard"),),
            face_match_plan=face_match,
            frame_provider=provider,
        )

        self.assertEqual(plan.eligible_count, 0)
        self.assertEqual(plan.analyzed_count, 0)
        self.assertEqual(
            plan.decisions[0].rejection_reason,
            "base_pan_crop_coordinate_space_mismatch",
        )

    def test_decode_failure_is_recorded_and_same_source_boundaries_are_ignored(self) -> None:
        _outgoing, _incoming, outgoing_box, incoming_box = _known_pair()
        timeline = _timeline(outgoing_box, incoming_box)
        failed = build_boundary_affine_plan(
            visual_timeline=timeline,
            sources=(Path("left.mp4"), Path("right.mp4")),
            transitions=(RenderTransition(tier="C", reason="hard"),),
            frame_provider=lambda _index, _seconds: None,
        )
        self.assertEqual(
            failed.decisions[0].rejection_reason,
            "boundary_frame_decode_failed",
        )

        same = VisualTimelinePlan(
            (
                timeline.choices[0],
                _choice(
                    1,
                    0,
                    master_start=1.0,
                    master_end=2.0,
                    source_start=4.0,
                    source_end=5.0,
                    face_box=incoming_box,
                ),
            ),
            1.0,
        )
        ignored = build_boundary_affine_plan(
            visual_timeline=same,
            sources=(Path("only.mp4"),),
            transitions=(RenderTransition(tier="D", reason="hold"),),
            frame_provider=lambda _index, _seconds: None,
        )
        self.assertEqual(ignored.decisions, ())

    def test_exact_cut_geometry_survives_background_affine_rejection(self) -> None:
        _outgoing, _incoming, outgoing_box, incoming_box = _known_pair()
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        exact_boxes = (
            (0.06, 0.18, 0.82, 0.72),
            (0.10, 0.20, 0.78, 0.68),
        )
        geometries = iter(
            BoundaryFrameSubjectGeometry(
                face_count=6,
                silhouette_box_xywh=box,
                silhouette_confidence=0.91,
            )
            for box in exact_boxes
        )

        plan = build_boundary_affine_plan(
            visual_timeline=_timeline(outgoing_box, incoming_box),
            sources=(Path("left.mp4"), Path("right.mp4")),
            transitions=(RenderTransition(tier="C", reason="hard"),),
            fps=30.0,
            output_width=640,
            output_height=360,
            frame_provider=lambda _source_index, _seconds: blank.copy(),
            geometry_provider=lambda frame: (
                next(geometries)
                if frame.shape == (360, 640, 3)
                else self.fail("geometry must receive the canonical output frame")
            ),
        )

        self.assertEqual(plan.analyzed_count, 1)
        self.assertEqual(plan.eligible_count, 0)
        self.assertEqual(plan.exact_subject_pair_count, 1)
        decision = plan.decisions[0]
        self.assertEqual(decision.rejection_reason, "descriptors_unavailable")
        assert decision.exact_subject_pair is not None
        self.assertEqual(decision.exact_subject_pair.kind, "silhouette")
        self.assertEqual(
            decision.exact_subject_pair.outgoing_box_xywh,
            exact_boxes[0],
        )
        self.assertEqual(
            decision.to_dict()["exact_subject_pair"]["incoming_box_xywh"],
            list(exact_boxes[1]),
        )


if __name__ == "__main__":
    unittest.main()
