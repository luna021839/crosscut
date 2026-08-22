from __future__ import annotations

from dataclasses import replace
import unittest

from crosscut.face_match import (
    FaceMatchConfig,
    _alignment_attempt,
    build_face_match_plan,
)
from crosscut.timeline import ShotType
from crosscut.visual_score import VisualTimelineChoice, VisualTimelinePlan


def _choice(
    segment_index: int,
    *,
    face_height: float = 0.12,
    center_x: float = 0.5,
    center_y: float = 0.42,
    failure: str | None = None,
) -> VisualTimelineChoice:
    width = face_height * 0.82
    box = (
        center_x - width / 2.0,
        center_y - face_height / 2.0,
        width,
        face_height,
    )
    return VisualTimelineChoice(
        segment_index=segment_index,
        source_index=segment_index % 2,
        canonical_master_start=float(segment_index),
        canonical_master_end=float(segment_index + 1),
        source_local_start=float(segment_index),
        source_local_end=float(segment_index + 1),
        quality_score=0.8,
        composition_score=0.8,
        boundary_score=0.8,
        shot_scale=0.4,
        shot_type=ShotType.MEDIUM,
        shot_id=f"shot-{segment_index}",
        face_samples_xywh=(box, box, box),
        face_match_candidate_failure=failure,
    )


def _with_silhouette(
    choice: VisualTimelineChoice,
    *,
    height: float,
) -> VisualTimelineChoice:
    width = min(0.82, height * 0.90)
    box = (0.5 - width / 2.0, 0.5 - height / 2.0, width, height)
    return replace(
        choice,
        silhouette_samples_xywh=(box, box, box),
        silhouette_centroid_samples_xy=((0.5, 0.5),) * 3,
        silhouette_backends=("opencv_zoo_pphumanseg_2023mar",) * 3,
        silhouette_confidences=(0.94, 0.94, 0.94),
    )


class FaceMatchTests(unittest.TestCase):
    def test_target_search_rejects_silhouette_wider_than_safe_crop(self) -> None:
        narrow_box = (0.10, 0.25, 0.80, 0.50)
        wide_box = (0.02, 0.25, 0.96, 0.50)
        choice = replace(
            _choice(0, failure="ambiguous_multiple_faces"),
            silhouette_samples_xywh=(narrow_box, narrow_box, wide_box),
            silhouette_centroid_samples_xy=((0.5, 0.5),) * 3,
            silhouette_backends=("opencv_zoo_pphumanseg_2023mar",) * 3,
            silhouette_confidences=(0.95, 0.95, 0.95),
        )

        plan = build_face_match_plan(
            visual_timeline=VisualTimelinePlan(
                choices=(choice,),
                optimizer_score=1.0,
            ),
            source_aspect_ratios=(16 / 9,),
        )

        self.assertIsNone(plan.target_silhouette_height)
        self.assertEqual(plan.applied_count, 0)
        self.assertEqual(plan.raw_segment_count, 1)
        self.assertIsNone(plan.segments[0].pan_crop)

    def test_final_gate_rejects_wide_silhouette_clipped_at_1_18_zoom(self) -> None:
        narrow_box = (0.10, 0.25, 0.80, 0.50)
        wide_box = (0.02, 0.25, 0.96, 0.50)
        settings = FaceMatchConfig()

        attempt = _alignment_attempt(
            choice=_choice(0, failure="ambiguous_multiple_faces"),
            alignment_subject="silhouette",
            samples=(narrow_box, narrow_box, wide_box),
            centers=((0.5, 0.5),) * 3,
            alignment_target=0.59,
            face_target=None,
            silhouette_target=0.59,
            source_aspect=16 / 9,
            target_screen_y=settings.silhouette_target_screen_y,
            maximum_zoom=settings.maximum_silhouette_zoom,
            settings=settings,
        )

        self.assertFalse(attempt.applied)
        self.assertAlmostEqual(attempt.required_zoom or 0.0, 1.18)
        self.assertEqual(attempt.fallback_reason, "silhouette_bbox_near_crop_edge")
        self.assertIsNone(attempt.pan_crop)

    def test_target_search_maximizes_safe_alignment_before_zoom_cost(self) -> None:
        timeline = VisualTimelinePlan(
            choices=(
                _choice(0, face_height=0.10),
                _choice(1, face_height=0.12),
                _choice(2, face_height=0.14),
            ),
            optimizer_score=1.0,
        )

        plan = build_face_match_plan(
            visual_timeline=timeline,
            source_aspect_ratios=(16 / 9, 16 / 9),
        )

        self.assertEqual(plan.applied_count, 3)
        self.assertEqual(plan.eligible_face_segment_count, 3)
        self.assertGreater(plan.target_candidate_count, 100)
        self.assertEqual(
            plan.target_selection_policy,
            "deterministic_max_safe_coverage_then_scale_zoom",
        )
        assert plan.target_face_height is not None
        self.assertGreaterEqual(plan.target_face_height, 0.94 * 0.14 - 1e-9)
        self.assertLessEqual(plan.target_face_height, 1.35 * 0.10 + 1e-9)

    def test_silhouette_target_search_avoids_percentile_raw_fallbacks(self) -> None:
        timeline = VisualTimelinePlan(
            choices=tuple(
                _with_silhouette(
                    _choice(index, failure="ambiguous_multiple_faces"),
                    height=height,
                )
                for index, height in enumerate((0.50, 0.58, 0.62))
            ),
            optimizer_score=1.0,
        )

        plan = build_face_match_plan(
            visual_timeline=timeline,
            source_aspect_ratios=(16 / 9, 16 / 9),
        )

        self.assertEqual(plan.silhouette_applied_count, 3)
        self.assertEqual(plan.eligible_silhouette_segment_count, 3)
        self.assertGreater(plan.silhouette_target_candidate_count, 100)
        assert plan.target_silhouette_height is not None
        self.assertGreaterEqual(
            plan.target_silhouette_height,
            0.94 * 0.62 - 1e-9,
        )
        self.assertLessEqual(
            plan.target_silhouette_height,
            1.18 * 0.50 + 1e-9,
        )

    def test_unsafe_face_scale_retries_safe_silhouette_before_raw_cut(self) -> None:
        choice = _with_silhouette(_choice(0, face_height=0.10), height=0.62)
        plan = build_face_match_plan(
            visual_timeline=VisualTimelinePlan(
                choices=(choice,),
                optimizer_score=1.0,
            ),
            source_aspect_ratios=(16 / 9,),
            config=FaceMatchConfig(target_face_height_override=0.20),
        )

        self.assertEqual(plan.applied_count, 1)
        self.assertEqual(plan.segments[0].alignment_subject, "silhouette")
        self.assertLessEqual(plan.segments[0].applied_zoom or 99.0, 1.18)

    def test_smaller_faces_get_conservative_aspect_preserving_pan_crop(self) -> None:
        timeline = VisualTimelinePlan(
            choices=(_choice(0, face_height=0.12), _choice(1, face_height=0.15)),
            optimizer_score=1.0,
        )
        plan = build_face_match_plan(
            visual_timeline=timeline,
            source_aspect_ratios=(16 / 9, 16 / 9),
            config=FaceMatchConfig(target_face_height_override=0.15),
        )

        self.assertEqual(plan.applied_count, 2)
        self.assertEqual(plan.fallback_count, 0)
        self.assertEqual(plan.aligned_segment_count, 2)
        self.assertEqual(plan.raw_segment_count, 0)
        diagnostics = plan.to_dict()
        self.assertEqual(diagnostics["count_basis"], "selected_segments")
        self.assertEqual(diagnostics["aligned_segment_count"], 2)
        self.assertNotIn("aligned_hard_cut_count", diagnostics)
        self.assertNotIn("raw_hard_cut_count", diagnostics)
        first = plan.segments[0]
        self.assertEqual(
            first.transition_tier,
            "moderate_confidence_aligned_hard_cut",
        )
        assert first.pan_crop is not None
        crop = first.pan_crop.keyframes[0].crop
        self.assertAlmostEqual(crop.width / crop.height, 1.0, places=6)
        self.assertLessEqual(first.applied_zoom or 99.0, 1.35)
        sample = timeline.choices[0].face_samples_xywh[0]
        assert sample is not None
        face_center_x = sample[0] + sample[2] / 2.0
        face_center_y = sample[1] + sample[3] / 2.0
        self.assertAlmostEqual((face_center_x - crop.x) / crop.width, 0.5)
        self.assertAlmostEqual((face_center_y - crop.y) / crop.height, 0.42)

    def test_ambiguous_face_uses_raw_frame_fallback(self) -> None:
        timeline = VisualTimelinePlan(
            choices=(_choice(0, failure="ambiguous_multiple_faces"),),
            optimizer_score=1.0,
        )
        plan = build_face_match_plan(
            visual_timeline=timeline,
            source_aspect_ratios=(16 / 9,),
        )

        self.assertEqual(plan.applied_count, 0)
        self.assertEqual(plan.raw_segment_count, 1)
        self.assertEqual(plan.segments[0].fallback_reason, "ambiguous_multiple_faces")
        self.assertEqual(
            plan.segments[0].to_dict()["transition_tier"],
            "raw_hard_cut_fallback",
        )
        self.assertIsNone(plan.segments[0].pan_crop)

    def test_ambiguous_face_can_use_stable_high_confidence_group_silhouette(self) -> None:
        choice = replace(
            _choice(0, failure="ambiguous_multiple_faces"),
            silhouette_samples_xywh=(
                (0.18, 0.16, 0.64, 0.62),
                (0.19, 0.16, 0.63, 0.61),
                (0.18, 0.17, 0.64, 0.62),
            ),
            silhouette_centroid_samples_xy=(
                (0.50, 0.49),
                (0.505, 0.495),
                (0.50, 0.50),
            ),
            silhouette_backends=("opencv_zoo_pphumanseg_2023mar",) * 3,
            silhouette_confidences=(0.94, 0.93, 0.95),
        )
        plan = build_face_match_plan(
            visual_timeline=VisualTimelinePlan(
                choices=(choice,),
                optimizer_score=1.0,
            ),
            source_aspect_ratios=(16 / 9,),
        )

        self.assertEqual(plan.applied_count, 1)
        self.assertEqual(plan.silhouette_applied_count, 1)
        segment = plan.segments[0]
        self.assertEqual(segment.alignment_subject, "silhouette")
        self.assertEqual(segment.transition_tier, "moderate_confidence_aligned_hard_cut")
        self.assertLessEqual(segment.applied_zoom or 99.0, 1.18)
        self.assertEqual(segment.to_dict()["mode"], "silhouette_match_pan_crop")

    def test_required_zoom_above_limit_uses_raw_frame(self) -> None:
        timeline = VisualTimelinePlan(
            choices=(_choice(0, face_height=0.08),),
            optimizer_score=1.0,
        )
        plan = build_face_match_plan(
            visual_timeline=timeline,
            source_aspect_ratios=(16 / 9,),
            config=FaceMatchConfig(target_face_height_override=0.15),
        )

        self.assertEqual(plan.applied_count, 0)
        self.assertEqual(
            plan.segments[0].fallback_reason,
            "required_zoom_exceeds_limit",
        )
        self.assertGreater(plan.segments[0].required_zoom or 0.0, 1.35)

    def test_edge_clamping_that_cannot_align_face_falls_back(self) -> None:
        timeline = VisualTimelinePlan(
            choices=(_choice(0, center_x=0.08, center_y=0.08),),
            optimizer_score=1.0,
        )
        plan = build_face_match_plan(
            visual_timeline=timeline,
            source_aspect_ratios=(16 / 9,),
            config=FaceMatchConfig(target_face_height_override=0.15),
        )

        self.assertEqual(plan.applied_count, 0)
        self.assertEqual(
            plan.segments[0].fallback_reason,
            "frame_edge_prevents_safe_alignment",
        )


if __name__ == "__main__":
    unittest.main()
