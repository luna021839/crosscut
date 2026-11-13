from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from crosscut.direct_member_rendered_continuity import (
    DirectMemberRenderedContinuityConfig,
    _recover_actual_pose_under_photometric_variants,
    evaluate_direct_member_rendered_continuity,
)
from crosscut.direct_member_target_face import FocalFaceCandidate
from crosscut.timeline import ShotType
from crosscut.visual_score import FrameVisualDescriptor


def _descriptor(
    *,
    position: float,
    pose: float,
    scale: float = 0.20,
    face_position: float | None = None,
    include_face: bool = True,
    include_pose: bool = True,
    silhouette_embedding: tuple[float, ...] | None = None,
) -> FrameVisualDescriptor:
    box = (position, 0.20, scale, 0.48)
    face_box = (
        (
            (face_position if face_position is not None else position),
            0.20,
            scale,
            0.48,
        )
        if include_face
        else None
    )
    embedding = (pose, pose * 0.5, 1.0 - pose * 0.25)
    return FrameVisualDescriptor(
        quality_score=0.95,
        composition_score=0.95,
        shot_scale=0.50,
        shot_type=ShotType.MEDIUM,
        face_count=1 if include_face else 0,
        face_detector_backend="yunet",
        structure_embedding=embedding,
        primary_face_bbox=face_box,
        pose_embedding=embedding if include_pose else (),
        pose_confidence=0.95 if include_pose else 0.0,
        pose_backend=(
            "synthetic_verified_pose" if include_pose else "none"
        ),
        detected_person_count=1,
        silhouette_embedding=(silhouette_embedding or embedding),
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


class _FocalProvider:
    def __init__(
        self,
        descriptors: dict[int, FrameVisualDescriptor],
        candidates: dict[int, tuple[FocalFaceCandidate, ...]],
    ) -> None:
        self.descriptors = descriptors
        self.candidates = candidates

    def __call__(self, seconds: float) -> FrameVisualDescriptor | None:
        return self.descriptors.get(int(round(seconds * 10.0)))

    def focal_face_candidates(
        self,
        seconds: float,
    ) -> tuple[FocalFaceCandidate, ...]:
        return self.candidates.get(int(round(seconds * 10.0)), ())


class DirectMemberRenderedContinuityTests(unittest.TestCase):
    def test_photometric_pose_recovery_preserves_original_review_geometry(
        self,
    ) -> None:
        original = _descriptor(
            position=0.30,
            pose=0.20,
            include_pose=False,
        )
        recovered_pose = _descriptor(
            position=0.70,
            pose=0.42,
            include_pose=True,
        )
        calls = 0

        def describer(_frame, **kwargs):
            nonlocal calls
            calls += 1
            self.assertTrue(kwargs["include_subject_geometry"])
            self.assertTrue(kwargs["include_formation_pose"])
            return original if calls == 1 else recovered_pose

        recovered, variant = _recover_actual_pose_under_photometric_variants(
            np.zeros((64, 64, 3), dtype=np.uint8),
            original,
            describer=describer,
        )

        self.assertEqual(calls, 2)
        self.assertEqual(variant, "neutral_luminance_clahe")
        self.assertEqual(recovered.pose_embedding, recovered_pose.pose_embedding)
        self.assertEqual(recovered.pose_backend, recovered_pose.pose_backend)
        self.assertEqual(recovered.primary_face_bbox, original.primary_face_bbox)
        self.assertEqual(
            recovered.silhouette_bbox_xywh,
            original.silhouette_bbox_xywh,
        )
        self.assertEqual(recovered.quality_score, original.quality_score)

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

    def test_boundary_uses_one_common_attention_basis(self) -> None:
        descriptors = {
            2: _descriptor(position=0.29, face_position=0.08, pose=0.20),
            3: _descriptor(position=0.30, face_position=0.09, pose=0.25),
            4: _descriptor(position=0.31, include_face=False, pose=0.30),
            5: _descriptor(position=0.32, include_face=False, pose=0.35),
        }

        result = evaluate_direct_member_rendered_continuity(
            Path("synthetic-member.mp4"),
            _plan(),
            descriptor_provider=_provider(descriptors),
        )

        self.assertTrue(result.passed, result.failure_reasons)
        self.assertEqual(
            result.boundaries[0]["attention_basis"],
            {
                "outgoing": "aggregate_human_silhouette",
                "incoming": "aggregate_human_silhouette",
            },
        )
        self.assertLess(
            result.boundaries[0]["attention_position_mismatch"],
            0.10,
        )

    def test_official_fancam_focal_face_overrides_group_silhouette(self) -> None:
        descriptors = {
            2: _descriptor(position=0.08, pose=0.20),
            3: _descriptor(position=0.09, pose=0.25),
            4: _descriptor(position=0.66, pose=0.30),
            5: _descriptor(position=0.67, pose=0.35),
        }
        candidates = {
            frame: (
                FocalFaceCandidate(
                    (0.36 + 0.01 * (frame - 2), 0.28, 0.12, 0.10),
                    (1.0, 0.04 * (frame - 2), 0.0),
                ),
                FocalFaceCandidate(
                    (0.80, 0.24, 0.10, 0.09),
                    (0.0, 1.0, 0.0),
                ),
            )
            for frame in (2, 3, 4, 5)
        }

        result = evaluate_direct_member_rendered_continuity(
            Path("synthetic-member.mp4"),
            _plan(),
            descriptor_provider=_FocalProvider(descriptors, candidates),
        )

        self.assertTrue(result.passed, result.failure_reasons)
        self.assertEqual(
            result.boundaries[0]["attention_basis"],
            {
                "outgoing": (
                    "verified_official_fancam_focal_face_correspondence"
                ),
                "incoming": (
                    "verified_official_fancam_focal_face_correspondence"
                ),
            },
        )
        self.assertEqual(
            result.metrics["focal_face_correspondence_boundary_count"],
            1,
        )
        self.assertFalse(
            result.boundaries[0]["focal_face_reference_guided"]
        )
        self.assertEqual(
            result.boundaries[0]["attention_geometry"]["outgoing"][
                "bbox_xywh"
            ],
            [0.37, 0.28, 0.12, 0.10],
        )

    def test_source_bound_references_keep_review_on_planned_performer(
        self,
    ) -> None:
        descriptors = {
            frame: _descriptor(position=0.08, pose=0.20 + frame * 0.01)
            for frame in (2, 3, 4, 5)
        }
        target_left = (1.0, 0.0, 0.0)
        target_right = (0.8, 0.6, 0.0)
        distractor = (0.0, 1.0, 0.0)
        candidates = {
            2: (
                FocalFaceCandidate((0.29, 0.28, 0.12, 0.10), target_left),
                FocalFaceCandidate((0.06, 0.24, 0.10, 0.09), distractor),
            ),
            3: (
                FocalFaceCandidate((0.30, 0.28, 0.12, 0.10), target_left),
                FocalFaceCandidate((0.07, 0.24, 0.10, 0.09), distractor),
            ),
            4: (
                FocalFaceCandidate((0.31, 0.28, 0.12, 0.10), target_right),
                FocalFaceCandidate((0.79, 0.24, 0.10, 0.09), distractor),
            ),
            5: (
                FocalFaceCandidate((0.32, 0.28, 0.12, 0.10), target_right),
                FocalFaceCandidate((0.80, 0.24, 0.10, 0.09), distractor),
            ),
        }

        result = evaluate_direct_member_rendered_continuity(
            Path("synthetic-member.mp4"),
            _plan(),
            descriptor_provider=_FocalProvider(descriptors, candidates),
            focal_reference_candidates_by_boundary_index={
                0: (
                    FocalFaceCandidate(
                        (0.30, 0.28, 0.12, 0.10),
                        target_left,
                    ),
                    FocalFaceCandidate(
                        (0.31, 0.28, 0.12, 0.10),
                        target_right,
                    ),
                ),
            },
        )

        self.assertTrue(result.passed, result.failure_reasons)
        self.assertTrue(result.boundaries[0]["focal_face_reference_guided"])
        self.assertEqual(
            result.metrics["focal_face_reference_guided_boundary_count"],
            1,
        )

    def test_focal_landmark_scale_ignores_color_sensitive_bbox_padding(
        self,
    ) -> None:
        descriptors = {
            frame: _descriptor(
                position=0.30,
                pose=0.20 + 0.01 * frame,
            )
            for frame in (2, 3, 4, 5)
        }
        target = (1.0, 0.0, 0.0)
        candidates = {
            2: (
                FocalFaceCandidate(
                    (0.35, 0.28, 0.10, 0.10),
                    target,
                    landmark_scale=0.045,
                ),
            ),
            3: (
                FocalFaceCandidate(
                    (0.35, 0.28, 0.10, 0.10),
                    target,
                    landmark_scale=0.045,
                ),
            ),
            4: (
                FocalFaceCandidate(
                    (0.32, 0.25, 0.16, 0.16),
                    target,
                    landmark_scale=0.045,
                ),
            ),
            5: (
                FocalFaceCandidate(
                    (0.32, 0.25, 0.16, 0.16),
                    target,
                    landmark_scale=0.045,
                ),
            ),
        }

        result = evaluate_direct_member_rendered_continuity(
            Path("synthetic-member.mp4"),
            _plan(),
            descriptor_provider=_FocalProvider(descriptors, candidates),
            focal_reference_candidates_by_boundary_index={
                0: (candidates[3][0], candidates[4][0]),
            },
        )

        self.assertTrue(result.passed, result.failure_reasons)
        boundary = result.boundaries[0]
        self.assertEqual(boundary["attention_scale_mismatch"], 0.0)
        self.assertEqual(
            boundary["attention_geometry"]["outgoing"]["scale_measurement"],
            "landmark_configuration_rms_squared",
        )
        self.assertEqual(
            boundary["attention_geometry"]["incoming"]["scale_area_proxy"],
            0.045**2,
        )

    def test_source_bound_face_does_not_fall_back_to_group_silhouette(
        self,
    ) -> None:
        descriptors = {
            2: _descriptor(
                position=0.10,
                pose=0.20,
                include_pose=False,
                silhouette_embedding=(1.0, 0.0, 0.0),
            ),
            3: _descriptor(
                position=0.10,
                pose=0.20,
                include_pose=False,
                silhouette_embedding=(1.0, 0.0, 0.0),
            ),
            4: _descriptor(
                position=0.70,
                pose=0.20,
                include_pose=False,
                silhouette_embedding=(0.0, 1.0, 0.0),
            ),
            5: _descriptor(
                position=0.70,
                pose=0.20,
                include_pose=False,
                silhouette_embedding=(0.0, 1.0, 0.0),
            ),
        }
        target = (1.0, 0.0, 0.0)
        candidates = {
            frame: (
                FocalFaceCandidate(
                    (0.30 + 0.005 * (frame - 2), 0.28, 0.10, 0.10),
                    target,
                ),
            )
            for frame in (2, 3, 4, 5)
        }

        result = evaluate_direct_member_rendered_continuity(
            Path("synthetic-member.mp4"),
            _plan(),
            config=DirectMemberRenderedContinuityConfig(
                minimum_actual_pose_coverage_fraction=0.0,
            ),
            descriptor_provider=_FocalProvider(descriptors, candidates),
            focal_reference_candidates_by_boundary_index={
                0: (candidates[3][0], candidates[4][0]),
            },
        )

        self.assertTrue(result.passed, result.failure_reasons)
        self.assertIsNone(result.boundaries[0]["pose_mismatch"])
        self.assertIsNone(result.boundaries[0]["silhouette_mismatch"])
        self.assertGreater(
            result.boundaries[0][
                "aggregate_silhouette_mismatch_diagnostic_only"
            ],
            0.42,
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

    def test_weak_aggregate_motion_does_not_trigger_screen_speed_gate(self) -> None:
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

        self.assertTrue(result.passed, result.failure_reasons)
        self.assertNotIn(
            "rendered_boundary_motion_speed_discontinuity",
            result.failure_reasons,
        )
        self.assertFalse(
            result.boundaries[0][
                "strong_output_subject_motion_gate_applied"
            ]
        )

    def test_strong_reference_guided_subject_speed_jump_fails_closed(
        self,
    ) -> None:
        descriptors = {
            frame: _descriptor(
                position=0.30,
                pose=0.20 + 0.01 * (frame - 2),
            )
            for frame in (2, 3, 4, 5)
        }
        target = (1.0, 0.0, 0.0)
        x_by_frame = {2: 0.10, 3: 0.25, 4: 0.30, 5: 0.75}
        candidates = {
            frame: (
                FocalFaceCandidate(
                    (x_by_frame[frame], 0.28, 0.10, 0.10),
                    target,
                ),
            )
            for frame in (2, 3, 4, 5)
        }

        result = evaluate_direct_member_rendered_continuity(
            Path("synthetic-member.mp4"),
            _plan(),
            descriptor_provider=_FocalProvider(descriptors, candidates),
            focal_reference_candidates_by_boundary_index={
                0: (
                    candidates[3][0],
                    candidates[4][0],
                ),
            },
        )

        self.assertFalse(result.passed)
        self.assertIn(
            "rendered_boundary_motion_speed_discontinuity",
            result.failure_reasons,
        )
        boundary = result.boundaries[0]
        self.assertTrue(
            boundary["strong_output_subject_motion_gate_applied"]
        )
        self.assertGreater(
            boundary["motion"]["subject_direction_cosine"],
            0.90,
        )

    def test_reference_authority_ignores_group_silhouette_when_pose_exists(
        self,
    ) -> None:
        plan = {
            "fps": 10.0,
            "output_frame_count": 20,
            "segments": [
                {"source_index": index, "duration": 0.4}
                for index in range(5)
            ],
            "transitions": [
                {"kind": "hard_cut", "frame_count": 0}
                for _ in range(4)
            ],
        }
        descriptors = {
            frame: _descriptor(
                position=0.30,
                pose=0.20 + 0.002 * frame,
                include_face=frame < 14,
                silhouette_embedding=(
                    (1.0, 0.0, 0.0)
                    if frame < 16
                    else (0.0, 1.0, 0.0)
                ),
            )
            for frame in range(2, 18)
        }
        target = (1.0, 0.0, 0.0)
        candidates = {
            frame: (
                FocalFaceCandidate(
                    (0.30 + 0.001 * frame, 0.28, 0.10, 0.10),
                    target,
                ),
            )
            for frame in range(2, 14)
        }
        reference = FocalFaceCandidate(
            (0.31, 0.28, 0.10, 0.10),
            target,
        )

        result = evaluate_direct_member_rendered_continuity(
            Path("synthetic-member.mp4"),
            plan,
            descriptor_provider=_FocalProvider(descriptors, candidates),
            focal_reference_candidates_by_boundary_index={
                boundary_index: (reference, reference)
                for boundary_index in range(4)
            },
        )

        self.assertTrue(result.passed, result.failure_reasons)
        self.assertEqual(
            result.metrics["focal_face_reference_guided_fraction"],
            0.75,
        )
        wide_boundary = result.boundaries[3]
        self.assertFalse(wide_boundary["focal_face_reference_guided"])
        self.assertIsNone(wide_boundary["silhouette_mismatch"])
        self.assertGreater(
            wide_boundary[
                "aggregate_silhouette_mismatch_diagnostic_only"
            ],
            0.42,
        )

    def test_reference_guided_coverage_below_floor_fails_closed(self) -> None:
        descriptors = {
            frame: _descriptor(
                position=0.30,
                pose=0.20 + 0.01 * frame,
                include_face=False,
            )
            for frame in (2, 3, 4, 5)
        }
        reference = FocalFaceCandidate(
            (0.30, 0.28, 0.10, 0.10),
            (1.0, 0.0, 0.0),
        )

        result = evaluate_direct_member_rendered_continuity(
            Path("synthetic-member.mp4"),
            _plan(),
            descriptor_provider=_FocalProvider(descriptors, {}),
            focal_reference_candidates_by_boundary_index={
                0: (reference, reference),
            },
        )

        self.assertFalse(result.passed)
        self.assertIn(
            "rendered_focal_reference_coverage_below_floor",
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

    def test_overlap_effect_persists_pure_focal_endpoints_for_camera_repair(
        self,
    ) -> None:
        descriptors = {
            3: _descriptor(position=0.20, pose=0.20),
            4: _descriptor(position=0.21, pose=0.22),
            6: _descriptor(position=0.59, pose=0.24),
            7: _descriptor(position=0.60, pose=0.26),
        }
        target = (1.0, 0.0, 0.0)
        candidates = {
            2: (FocalFaceCandidate((0.19, 0.28, 0.10, 0.10), target),),
            3: (FocalFaceCandidate((0.20, 0.28, 0.10, 0.10), target),),
            7: (FocalFaceCandidate((0.60, 0.28, 0.10, 0.10), target),),
        }

        result = evaluate_direct_member_rendered_continuity(
            Path("synthetic-member.mp4"),
            _effect_plan(),
            descriptor_provider=_FocalProvider(descriptors, candidates),
            focal_reference_candidates_by_boundary_index={
                0: (candidates[3][0], candidates[7][0]),
            },
        )

        boundary = result.boundaries[0]
        self.assertEqual(
            boundary["pure_endpoint_sample_frame_indices"],
            [3, 7],
        )
        self.assertTrue(boundary["focal_face_reference_guided"])
        for endpoint in boundary['endpoint_steps']:
            self.assertFalse(endpoint['target_reference_guided'])
            self.assertEqual(endpoint['attention_basis'],
                {'left':'unambiguous_dominant_face_bbox_geometry_only',
                 'right':'unambiguous_dominant_face_bbox_geometry_only'})
            self.assertIsNotNone(endpoint['attention_geometry']['left'])
            self.assertIsNotNone(endpoint['attention_geometry']['right'])
        self.assertGreater(boundary["attention_position_mismatch"], 0.90)
        self.assertEqual(
            result.metrics[
                "effect_focal_face_reference_guided_boundary_count"
            ],
            1,
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

    def test_effect_group_mask_scale_failure_reports_actual_geometry_not_target_claim(self):
        descriptors={n:_descriptor(position=.3,pose=.2,scale=(.2 if n<7 else .4),
                                   include_face=False) for n in (3,4,6,7)}
        result=evaluate_direct_member_rendered_continuity(Path('synthetic-member.mp4'),
            _effect_plan(),descriptor_provider=_provider(descriptors))
        self.assertIn('rendered_effect_endpoint_scale_discontinuity',result.failure_reasons)
        row=result.boundaries[0]['endpoint_steps'][1]
        self.assertEqual(row['attention_basis'],
            {'left':'aggregate_human_silhouette','right':'aggregate_human_silhouette'})
        self.assertFalse(row['target_reference_guided'])
        self.assertEqual(row['attention_geometry']['left']['bbox_xywh'],[.3,.2,.2,.48])
        self.assertEqual(row['attention_geometry']['right']['bbox_xywh'],[.3,.2,.4,.48])


if __name__ == "__main__":
    unittest.main()
