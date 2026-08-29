from __future__ import annotations

from dataclasses import replace
import unittest

from crosscut.face_match import (
    FaceMatchConfig,
    _alignment_attempt,
    build_face_match_plan,
)
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
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
    def test_chorus_single_subject_wide_uses_bounded_nonwide_rescue(self) -> None:
        close = replace(
            _choice(0, face_height=0.40),
            shot_type=ShotType.CLOSE_UP,
        )
        distant = replace(
            _choice(1, face_height=0.08),
            shot_type=ShotType.WIDE,
            entry_shot_type=ShotType.WIDE,
            exit_shot_type=ShotType.WIDE,
            detected_person_counts=(1, 1, 1),
            formation_person_counts=(1, 1, 1),
        )
        section_plan = SectionPlan(
            sections=(
                SectionDirective(
                    section_index=0,
                    role=SectionRole.CHORUS_LIKE,
                    start_seconds=0.0,
                    end_seconds=2.0,
                    energy_score=0.9,
                    confidence=0.9,
                    phrase_count=1,
                    preferred_source_hold_beats=2,
                    target_source_switches_per_minute=18.0,
                    camera_motion_budget=0.8,
                    target_roll_events_per_minute=0.0,
                    establish_with_wide_shot=False,
                    protect_last_beat_from_extra_cut=False,
                ),
            )
        )

        plan = build_face_match_plan(
            visual_timeline=VisualTimelinePlan(
                choices=(close, distant),
                optimizer_score=1.0,
            ),
            source_aspect_ratios=(16 / 9, 16 / 9),
            section_plan=section_plan,
        )

        rescue = plan.segments[1]
        self.assertTrue(rescue.applied)
        self.assertEqual(
            rescue.alignment_purpose,
            "chorus_single_subject_nonwide_rescue",
        )
        self.assertEqual(plan.nonwide_rescue_applied_count, 1)
        self.assertGreaterEqual(
            0.08 * (rescue.applied_zoom or 0.0),
            FaceMatchConfig().minimum_nonwide_rescue_face_height,
        )
        self.assertLessEqual(rescue.applied_zoom or 99.0, 1.35)

    def test_chorus_formation_wide_cannot_use_single_subject_rescue(self) -> None:
        close = replace(
            _choice(0, face_height=0.40),
            shot_type=ShotType.CLOSE_UP,
        )
        formation = replace(
            _choice(1, face_height=0.08),
            shot_type=ShotType.GROUP_WIDE,
            entry_shot_type=ShotType.GROUP_WIDE,
            exit_shot_type=ShotType.GROUP_WIDE,
            detected_person_counts=(4, 4, 4),
            formation_person_counts=(4, 4, 4),
        )
        section_plan = SectionPlan(
            sections=(
                SectionDirective(
                    section_index=0,
                    role=SectionRole.CHORUS_LIKE,
                    start_seconds=0.0,
                    end_seconds=2.0,
                    energy_score=0.9,
                    confidence=0.9,
                    phrase_count=1,
                    preferred_source_hold_beats=2,
                    target_source_switches_per_minute=18.0,
                    camera_motion_budget=0.8,
                    target_roll_events_per_minute=0.0,
                    establish_with_wide_shot=True,
                    protect_last_beat_from_extra_cut=False,
                ),
            )
        )

        plan = build_face_match_plan(
            visual_timeline=VisualTimelinePlan(
                choices=(close, formation),
                optimizer_score=1.0,
            ),
            source_aspect_ratios=(16 / 9, 16 / 9),
            section_plan=section_plan,
        )

        self.assertFalse(plan.segments[1].applied)
        self.assertEqual(plan.nonwide_rescue_applied_count, 0)

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
        self.assertEqual(len(first.pan_crop.keyframes), 1)
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

    def test_cross_source_one_sided_crop_remains_a_local_hard_cut_candidate(self) -> None:
        timeline = VisualTimelinePlan(
            choices=(
                _choice(0, face_height=0.13),
                _choice(
                    1,
                    face_height=0.13,
                    failure="ambiguous_multiple_faces",
                ),
            ),
            optimizer_score=1.0,
        )

        plan = build_face_match_plan(
            visual_timeline=timeline,
            source_aspect_ratios=(16 / 9, 16 / 9),
            config=FaceMatchConfig(target_face_height_override=0.15),
        )

        self.assertEqual(plan.applied_count, 1)
        self.assertTrue(plan.segments[0].applied)
        self.assertIsNotNone(plan.segments[0].pan_crop)
        self.assertIsNone(plan.segments[0].fallback_reason)
        self.assertEqual(
            plan.segments[1].fallback_reason,
            "ambiguous_multiple_faces",
        )

    def test_cross_source_raw_neighbor_does_not_erase_safe_crop_chain(self) -> None:
        timeline = VisualTimelinePlan(
            choices=(
                _choice(0, face_height=0.13),
                _choice(1, face_height=0.13),
                _choice(
                    2,
                    face_height=0.13,
                    failure="ambiguous_multiple_faces",
                ),
            ),
            optimizer_score=1.0,
        )

        plan = build_face_match_plan(
            visual_timeline=timeline,
            source_aspect_ratios=(16 / 9, 16 / 9),
            config=FaceMatchConfig(target_face_height_override=0.15),
        )

        self.assertEqual(plan.applied_count, 2)
        self.assertEqual(
            tuple(segment.fallback_reason for segment in plan.segments),
            (
                None,
                None,
                "ambiguous_multiple_faces",
            ),
        )
        self.assertTrue(
            all(segment.pan_crop is not None for segment in plan.segments[:2])
        )
        self.assertIsNone(plan.segments[2].pan_crop)

    def test_two_of_three_face_samples_recover_fixed_zoom_pan(self) -> None:
        base = _choice(0, face_height=0.13)
        entry = base.face_samples_xywh[0]
        assert entry is not None
        exit_box = (entry[0] + 0.025, entry[1], entry[2], entry[3])
        partial = replace(
            base,
            face_samples_xywh=(entry, None, exit_box),
            face_match_candidate_failure="face_not_detected",
        )

        plan = build_face_match_plan(
            visual_timeline=VisualTimelinePlan(
                choices=(partial,),
                optimizer_score=1.0,
            ),
            source_aspect_ratios=(16 / 9,),
            config=FaceMatchConfig(target_face_height_override=0.15),
        )

        self.assertEqual(plan.applied_count, 1)
        segment = plan.segments[0]
        self.assertEqual(segment.alignment_subject, "face")
        assert segment.pan_crop is not None
        self.assertEqual(
            tuple(item.time_seconds for item in segment.pan_crop.keyframes),
            (0.0, 1.0),
        )
        first, last = segment.pan_crop.keyframes
        self.assertAlmostEqual(first.crop.width, last.crop.width)
        self.assertAlmostEqual(first.crop.height, last.crop.height)
        self.assertNotAlmostEqual(first.crop.x, last.crop.x)

    def test_missing_face_boundary_sample_cannot_authorize_crop(self) -> None:
        base = _choice(0, face_height=0.13)
        middle = base.face_samples_xywh[1]
        exit_box = base.face_samples_xywh[2]
        partial = replace(
            base,
            face_samples_xywh=(None, middle, exit_box),
            face_match_candidate_failure="face_not_detected",
        )

        plan = build_face_match_plan(
            visual_timeline=VisualTimelinePlan(
                choices=(partial,),
                optimizer_score=1.0,
            ),
            source_aspect_ratios=(16 / 9,),
            config=FaceMatchConfig(target_face_height_override=0.15),
        )

        self.assertEqual(plan.applied_count, 0)
        self.assertEqual(plan.segments[0].fallback_reason, "face_not_detected")

    def test_one_face_sample_cannot_authorize_crop(self) -> None:
        base = _choice(0, face_height=0.13)
        entry = base.face_samples_xywh[0]
        partial = replace(
            base,
            face_samples_xywh=(entry, None, None),
            face_match_candidate_failure="face_not_detected",
        )

        plan = build_face_match_plan(
            visual_timeline=VisualTimelinePlan(
                choices=(partial,),
                optimizer_score=1.0,
            ),
            source_aspect_ratios=(16 / 9,),
            config=FaceMatchConfig(target_face_height_override=0.15),
        )

        self.assertEqual(plan.applied_count, 0)
        self.assertEqual(plan.segments[0].fallback_reason, "face_not_detected")

    def test_two_of_three_silhouette_samples_recover_conservative_pan(self) -> None:
        base = _choice(0, failure="ambiguous_multiple_faces")
        box = (0.20, 0.18, 0.60, 0.62)
        partial = replace(
            base,
            silhouette_samples_xywh=(box, None, box),
            silhouette_centroid_samples_xy=((0.5, 0.49), None, (0.5, 0.50)),
            silhouette_backends=(
                "opencv_zoo_pphumanseg_2023mar",
                "none",
                "opencv_zoo_pphumanseg_2023mar",
            ),
            silhouette_confidences=(0.94, 0.0, 0.95),
        )

        plan = build_face_match_plan(
            visual_timeline=VisualTimelinePlan(
                choices=(partial,),
                optimizer_score=1.0,
            ),
            source_aspect_ratios=(16 / 9,),
        )

        self.assertEqual(plan.silhouette_applied_count, 1)
        self.assertEqual(plan.segments[0].alignment_subject, "silhouette")

    def test_missing_silhouette_boundary_sample_cannot_authorize_crop(self) -> None:
        base = _choice(0, failure="ambiguous_multiple_faces")
        box = (0.20, 0.18, 0.60, 0.62)
        partial = replace(
            base,
            silhouette_samples_xywh=(None, box, box),
            silhouette_centroid_samples_xy=(None, (0.5, 0.49), (0.5, 0.50)),
            silhouette_backends=(
                "none",
                "opencv_zoo_pphumanseg_2023mar",
                "opencv_zoo_pphumanseg_2023mar",
            ),
            silhouette_confidences=(0.0, 0.94, 0.95),
        )

        plan = build_face_match_plan(
            visual_timeline=VisualTimelinePlan(
                choices=(partial,),
                optimizer_score=1.0,
            ),
            source_aspect_ratios=(16 / 9,),
        )

        self.assertEqual(plan.applied_count, 0)
        self.assertEqual(
            plan.segments[0].fallback_reason,
            "ambiguous_multiple_faces",
        )

    def test_mixed_face_silhouette_boundary_demotes_only_silhouette_side(self) -> None:
        silhouette = _with_silhouette(
            _choice(1, failure="ambiguous_multiple_faces"),
            height=0.62,
        )
        plan = build_face_match_plan(
            visual_timeline=VisualTimelinePlan(
                choices=(_choice(0, face_height=0.13), silhouette),
                optimizer_score=1.0,
            ),
            source_aspect_ratios=(16 / 9, 16 / 9),
            config=FaceMatchConfig(target_face_height_override=0.15),
        )

        self.assertEqual(plan.applied_count, 1)
        self.assertTrue(plan.segments[0].applied)
        self.assertEqual(
            plan.segments[1].fallback_reason,
            "cross_source_alignment_subject_mismatch",
        )

    def test_continuous_shot_demotion_does_not_cross_visible_source_cut(self) -> None:
        choices = (
            replace(
                _choice(0, face_height=0.13),
                source_index=0,
                shot_id="source-0:shot-1",
            ),
            replace(
                _choice(1, failure="ambiguous_multiple_faces"),
                source_index=0,
                shot_id="source-0:shot-1",
            ),
            replace(_choice(2, face_height=0.13), source_index=1),
        )
        plan = build_face_match_plan(
            visual_timeline=VisualTimelinePlan(
                choices=choices,
                optimizer_score=1.0,
            ),
            source_aspect_ratios=(16 / 9, 16 / 9),
            config=FaceMatchConfig(target_face_height_override=0.15),
        )

        self.assertEqual(plan.applied_count, 1)
        self.assertFalse(plan.segments[0].applied)
        self.assertFalse(plan.segments[1].applied)
        self.assertTrue(plan.segments[2].applied)

    def test_continuous_source_hold_cannot_hide_raw_to_crop_punch_in(self) -> None:
        timeline = VisualTimelinePlan(
            choices=(
                replace(
                    _choice(0, face_height=0.13),
                    source_index=0,
                    shot_id="source-0:shot-1",
                ),
                replace(
                    _choice(
                        1,
                        face_height=0.13,
                        failure="ambiguous_multiple_faces",
                    ),
                    source_index=0,
                    shot_id="source-0:shot-1",
                ),
            ),
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
            "continuous_shot_crop_pair_incompatible",
        )
        self.assertIsNone(plan.segments[0].pan_crop)

    def test_continuous_source_hold_stitches_safe_crop_endpoints(self) -> None:
        choices = (
            replace(
                _choice(0, face_height=0.13, center_x=0.45),
                source_index=0,
                shot_id="source-0:shot-1",
            ),
            replace(
                _choice(1, face_height=0.13, center_x=0.55),
                source_index=0,
                shot_id="source-0:shot-1",
            ),
        )
        plan = build_face_match_plan(
            visual_timeline=VisualTimelinePlan(
                choices=choices,
                optimizer_score=1.0,
            ),
            source_aspect_ratios=(16 / 9,),
            config=FaceMatchConfig(target_face_height_override=0.15),
        )

        self.assertEqual(plan.applied_count, 2)
        self.assertEqual(plan.continuous_run_stitched_segment_count, 2)
        self.assertTrue(
            all(segment.continuous_run_stitched for segment in plan.segments)
        )
        left = plan.segments[0].pan_crop
        right = plan.segments[1].pan_crop
        assert left is not None and right is not None
        left_width, left_height = left.fixed_crop_size
        right_width, right_height = right.fixed_crop_size
        left_position = left.fixed_crop_positions()[-1]
        right_position = right.fixed_crop_positions()[0]
        self.assertEqual(
            (left_position[1], left_position[2], left_width, left_height),
            (right_position[1], right_position[2], right_width, right_height),
        )
        self.assertGreaterEqual(plan.segments[0].applied_zoom or 0.0, 1.025)

    def test_continuous_source_hold_stitches_subpercent_crop_reset(self) -> None:
        choices = (
            replace(
                _choice(0, face_height=0.13, center_x=0.500),
                source_index=0,
                shot_id="source-0:shot-1",
            ),
            replace(
                _choice(1, face_height=0.13, center_x=0.502),
                source_index=0,
                shot_id="source-0:shot-1",
            ),
        )

        plan = build_face_match_plan(
            visual_timeline=VisualTimelinePlan(
                choices=choices,
                optimizer_score=1.0,
            ),
            source_aspect_ratios=(16 / 9,),
            config=FaceMatchConfig(target_face_height_override=0.15),
        )

        self.assertEqual(plan.continuous_run_stitched_segment_count, 2)
        left = plan.segments[0].pan_crop
        right = plan.segments[1].pan_crop
        assert left is not None and right is not None
        self.assertEqual(
            left.fixed_crop_positions()[-1][1:],
            right.fixed_crop_positions()[0][1:],
        )

    def test_fast_invisible_pan_uses_static_covering_crop(self) -> None:
        duration = 0.6122383722162043
        choices = tuple(
            replace(
                _choice(
                    index,
                    face_height=0.10,
                    center_x=center_x,
                ),
                source_index=0,
                shot_id="source-0:shot-1",
                canonical_master_start=index * duration,
                canonical_master_end=(index + 1) * duration,
                source_local_start=index * duration,
                source_local_end=(index + 1) * duration,
            )
            for index, center_x in enumerate((0.43, 0.57))
        )

        plan = build_face_match_plan(
            visual_timeline=VisualTimelinePlan(
                choices=choices,
                optimizer_score=1.0,
            ),
            source_aspect_ratios=(16 / 9,),
            config=FaceMatchConfig(target_face_height_override=0.135),
        )

        self.assertEqual(plan.applied_count, 2)
        self.assertEqual(plan.continuous_run_stitched_segment_count, 2)
        tracks = tuple(item.pan_crop for item in plan.segments)
        self.assertTrue(all(track is not None for track in tracks))
        self.assertTrue(
            all(len(track.keyframes) == 1 for track in tracks if track is not None)
        )
        self.assertEqual(
            tracks[0].fixed_crop_positions() if tracks[0] is not None else (),
            tracks[1].fixed_crop_positions() if tracks[1] is not None else (),
        )

    def test_continuous_source_hold_preserves_matching_static_crop(self) -> None:
        choices = (
            replace(
                _choice(0, face_height=0.13),
                source_index=0,
                shot_id="source-0:shot-1",
            ),
            replace(
                _choice(1, face_height=0.13),
                source_index=0,
                shot_id="source-0:shot-1",
            ),
        )
        plan = build_face_match_plan(
            visual_timeline=VisualTimelinePlan(
                choices=choices,
                optimizer_score=1.0,
            ),
            source_aspect_ratios=(16 / 9,),
            config=FaceMatchConfig(target_face_height_override=0.15),
        )

        self.assertEqual(plan.applied_count, 2)
        self.assertTrue(all(segment.applied for segment in plan.segments))

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
