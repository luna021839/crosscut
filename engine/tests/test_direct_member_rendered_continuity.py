from __future__ import annotations

from pathlib import Path
import unittest

from crosscut.direct_member_rendered_continuity import (
    evaluate_direct_member_rendered_continuity,
)
from crosscut.timeline import ShotType
from crosscut.visual_score import FrameVisualDescriptor


def _descriptor(
    *,
    position: float,
    pose: float,
    scale: float = 0.20,
) -> FrameVisualDescriptor:
    box = (position, 0.20, scale, 0.48)
    embedding = (pose, pose * 0.5, 1.0 - pose * 0.25)
    return FrameVisualDescriptor(
        quality_score=0.95,
        composition_score=0.95,
        shot_scale=0.50,
        shot_type=ShotType.MEDIUM,
        face_count=1,
        face_detector_backend="yunet",
        structure_embedding=embedding,
        primary_face_bbox=box,
        pose_embedding=embedding,
        pose_confidence=0.95,
        pose_backend="synthetic_verified_pose",
        detected_person_count=1,
        silhouette_embedding=embedding,
        silhouette_bbox_xywh=box,
        silhouette_centroid_xy=(
            box[0] + box[2] / 2.0,
            box[1] + box[3] / 2.0,
        ),
        silhouette_area_ratio=box[2] * box[3],
        silhouette_confidence=0.95,
        silhouette_backend="synthetic_verified_silhouette",
    )


def _plan() -> dict[str, object]:
    return {
        "fps": 10.0,
        "output_frame_count": 8,
        "segments": [
            {"source_index": 0, "duration": 0.4},
            {"source_index": 1, "duration": 0.4},
        ],
        "transitions": [{"kind": "hard_cut", "frame_count": 0}],
    }


def _effect_plan() -> dict[str, object]:
    plan = _plan()
    plan["transitions"] = [
        {"kind": "face_dissolve", "frame_count": 3}
    ]
    return plan


def _provider(
    descriptors: dict[int, FrameVisualDescriptor],
    requests: list[int] | None = None,
):
    def provide(seconds: float) -> FrameVisualDescriptor | None:
        frame = int(round(seconds * 10.0))
        if requests is not None:
            requests.append(frame)
        return descriptors.get(frame)

    return provide


class DirectMemberRenderedContinuityTests(unittest.TestCase):
    def test_smooth_exact_rendered_boundary_passes(self) -> None:
        descriptors = {
            2: _descriptor(position=0.30, pose=0.20),
            3: _descriptor(position=0.31, pose=0.25),
            4: _descriptor(position=0.32, pose=0.30),
            5: _descriptor(position=0.33, pose=0.35),
        }

        result = evaluate_direct_member_rendered_continuity(
            Path("synthetic-member.mp4"),
            _plan(),
            descriptor_provider=_provider(descriptors),
        )

        self.assertTrue(result.passed, result.failure_reasons)
        self.assertEqual(result.metrics["hard_cut_source_switch_count"], 1)
        self.assertEqual(
            result.boundaries[0]["sample_frame_indices"],
            [2, 3, 4, 5],
        )
        self.assertTrue(
            result.boundaries[0]["motion"]["output_space_subject_motion_used"]
        )

    def test_evaluates_exact_incoming_first_visible_frame(self) -> None:
        requests: list[int] = []
        descriptors = {
            2: _descriptor(position=0.30, pose=0.20),
            3: _descriptor(position=0.31, pose=0.25),
            # The exact incoming frame is incompatible. Frame +1 deliberately
            # matches the outgoing side so the former off-by-one audit would
            # have accepted this cut.
            4: _descriptor(position=0.70, pose=0.90),
            5: _descriptor(position=0.32, pose=0.30),
        }

        result = evaluate_direct_member_rendered_continuity(
            Path("synthetic-member.mp4"),
            _plan(),
            descriptor_provider=_provider(descriptors, requests),
        )

        self.assertFalse(result.passed)
        self.assertEqual(requests, [2, 3, 4, 5])
        self.assertIn(
            "rendered_boundary_position_discontinuity",
            result.failure_reasons,
        )
        self.assertEqual(
            result.failure_boundary_indices[
                "rendered_boundary_position_discontinuity"
            ],
            (0,),
        )

    def test_pose_discontinuity_fails_closed(self) -> None:
        descriptors = {
            2: _descriptor(position=0.30, pose=0.10),
            3: _descriptor(position=0.31, pose=0.15),
            4: _descriptor(position=0.32, pose=0.95),
            5: _descriptor(position=0.33, pose=0.90),
        }

        result = evaluate_direct_member_rendered_continuity(
            Path("synthetic-member.mp4"),
            _plan(),
            descriptor_provider=_provider(descriptors),
        )

        self.assertFalse(result.passed)
        self.assertIn("rendered_boundary_pose_discontinuity", result.failure_reasons)

    def test_motion_speed_discontinuity_fails_closed(self) -> None:
        descriptors = {
            2: _descriptor(position=0.30, pose=0.10),
            3: _descriptor(position=0.31, pose=0.20),
            4: _descriptor(position=0.32, pose=0.21),
            5: _descriptor(position=0.52, pose=0.61),
        }

        result = evaluate_direct_member_rendered_continuity(
            Path("synthetic-member.mp4"),
            _plan(),
            descriptor_provider=_provider(descriptors),
        )

        self.assertFalse(result.passed)
        self.assertIn(
            "rendered_boundary_motion_speed_discontinuity",
            result.failure_reasons,
        )

    def test_overlap_effect_entry_and_exit_are_checked(self) -> None:
        requests: list[int] = []
        descriptors = {
            3: _descriptor(position=0.30, pose=0.20),
            4: _descriptor(position=0.31, pose=0.22),
            6: _descriptor(position=0.40, pose=0.40),
            7: _descriptor(position=0.41, pose=0.42),
        }

        result = evaluate_direct_member_rendered_continuity(
            Path("synthetic-member.mp4"),
            _effect_plan(),
            descriptor_provider=_provider(descriptors, requests),
        )

        self.assertTrue(result.passed, result.failure_reasons)
        self.assertEqual(requests, [3, 4, 6, 7])
        self.assertEqual(result.metrics["hard_cut_source_switch_count"], 0)
        self.assertEqual(result.metrics["effect_source_switch_count"], 1)
        self.assertEqual(result.metrics["effect_endpoint_pair_count"], 2)
        self.assertEqual(
            result.boundaries[0]["endpoint_steps"][1]["sample_frame_indices"],
            [6, 7],
        )

    def test_overlap_effect_exit_jump_fails_closed(self) -> None:
        descriptors = {
            3: _descriptor(position=0.30, pose=0.20),
            4: _descriptor(position=0.31, pose=0.22),
            6: _descriptor(position=0.40, pose=0.40),
            7: _descriptor(position=0.70, pose=0.95),
        }

        result = evaluate_direct_member_rendered_continuity(
            Path("synthetic-member.mp4"),
            _effect_plan(),
            descriptor_provider=_provider(descriptors),
        )

        self.assertFalse(result.passed)
        self.assertIn(
            "rendered_effect_endpoint_position_discontinuity",
            result.failure_reasons,
        )


if __name__ == "__main__":
    unittest.main()
