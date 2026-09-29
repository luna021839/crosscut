from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import unittest

from crosscut.boundary_affine_match import (
    BoundaryAffineMatchResult,
    BoundaryAffineTransformEvidence,
)
from crosscut.boundary_affine_plan import (
    BoundaryAffineDecision,
    BoundaryAffinePlan,
    BoundarySubjectPair,
)
from crosscut.editorial_motion import (
    EditorialMotionConfig,
    build_editorial_motion_plan,
)
from crosscut.face_match import FaceMatchPlan, FaceMatchSegment
from crosscut.render import (
    NormalizedCrop,
    PanCropKeyframe,
    PanCropTrack,
    RenderPlan,
    RenderSegment,
    RenderTransition,
    TransitionKind,
)
from crosscut.request import AspectRatio
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.timeline import ShotType
from crosscut.visual_score import VisualTimelineChoice, VisualTimelinePlan


def _choice(
    segment_index: int,
    *,
    source_index: int,
    start: float,
    end: float,
    shot_id: str,
    face_height: float | None = None,
    face_roll_degrees: float | None = None,
) -> VisualTimelineChoice:
    face = None
    if face_height is not None:
        face = (
            0.5 - face_height * 0.40,
            0.42 - face_height / 2.0,
            face_height * 0.80,
            face_height,
        )
    return VisualTimelineChoice(
        segment_index=segment_index,
        source_index=source_index,
        canonical_master_start=start,
        canonical_master_end=end,
        source_local_start=start,
        source_local_end=end,
        quality_score=0.90,
        composition_score=0.90,
        boundary_score=0.90,
        shot_scale=0.45,
        shot_type=ShotType.MEDIUM,
        shot_id=shot_id,
        entry_shot_scale=0.40,
        exit_shot_scale=0.48,
        face_samples_xywh=(face, face, face),
        face_detector_backends=("opencv_zoo_yunet_2023mar",) * 3,
        face_head_roll_samples_radians=(
            (
                math.radians(face_roll_degrees)
                if face_roll_degrees is not None
                else None
            ),
            None,
            None,
        ),
    )


def _slope(left_time: float, left_zoom: float, right_time: float, right_zoom: float) -> float:
    return (right_zoom - left_zoom) / (right_time - left_time)


def _face_match(
    segment_index: int,
    source_index: int,
    *,
    crop_height: float,
) -> FaceMatchSegment:
    crop_width = crop_height
    crop = PanCropTrack(
        (
            PanCropKeyframe(
                0.0,
                NormalizedCrop(
                    (1.0 - crop_width) / 2.0,
                    (1.0 - crop_height) / 2.0,
                    crop_width,
                    crop_height,
                ),
            ),
        )
    )
    return FaceMatchSegment(
        segment_index=segment_index,
        source_index=source_index,
        applied=True,
        fallback_reason=None,
        observed_face_height=0.1,
        target_face_height=0.12,
        required_zoom=1.0 / crop_height,
        applied_zoom=1.0 / crop_height,
        pan_crop=crop,
        alignment_subject="face",
    )


class EditorialMotionTests(unittest.TestCase):
    def test_eligible_pixel_affine_drives_incoming_settle_before_box_fallback(self) -> None:
        timeline = VisualTimelinePlan(
            choices=(
                _choice(
                    0,
                    source_index=0,
                    start=0.0,
                    end=1.0,
                    shot_id="left",
                    face_height=0.10,
                ),
                _choice(
                    1,
                    source_index=1,
                    start=1.0,
                    end=2.0,
                    shot_id="right",
                    face_height=0.11,
                ),
            ),
            optimizer_score=1.0,
        )
        transform = BoundaryAffineTransformEvidence(
            1.09,
            0.507,
            0.494,
            1.0,
        )
        matcher = BoundaryAffineMatchResult(
            True,
            None,
            transform,
            {
                "affine_inlier_ratio": 0.82,
                "subject_review": {"geometry_improvement": 0.09},
            },
        )
        affine_plan = BoundaryAffinePlan(
            (
                BoundaryAffineDecision(
                    0,
                    0,
                    1,
                    0.966,
                    1.0,
                    "face",
                    matcher,
                    None,
                ),
            )
        )

        plan = build_editorial_motion_plan(
            visual_timeline=timeline,
            transitions=(RenderTransition(tier="C", reason="hard"),),
            boundary_affine_plan=affine_plan,
        )

        self.assertEqual(plan.boundary_match_count, 1)
        self.assertEqual(plan.affine_boundary_match_count, 1)
        self.assertEqual(plan.roll_event_count, 0)
        self.assertEqual(dict(plan.authored_event_counts)["roll"], 1)
        incoming = plan.track_for(1)
        assert incoming is not None
        self.assertAlmostEqual(incoming.keyframes[0].zoom, 1.09)
        self.assertAlmostEqual(incoming.keyframes[0].roll_degrees, 1.0)
        self.assertAlmostEqual(incoming.final_keyframe.zoom, 1.0)
        self.assertIn("orb_affine_subject_matched_hard_cut", incoming.reason)

    def test_exact_cut_geometry_drives_safe_fallback_when_orb_background_fails(self) -> None:
        timeline = VisualTimelinePlan(
            choices=(
                _choice(
                    0,
                    source_index=0,
                    start=0.0,
                    end=1.0,
                    shot_id="left",
                ),
                _choice(
                    1,
                    source_index=1,
                    start=1.0,
                    end=2.0,
                    shot_id="right",
                ),
            ),
            optimizer_score=1.0,
        )
        rejected_match = BoundaryAffineMatchResult(
            False,
            "insufficient_ratio_matches",
            None,
            {"ratio_match_count": 3},
        )
        exact_pair = BoundarySubjectPair(
            "silhouette",
            (0.27, 0.20, 0.46, 0.60),
            (0.30, 0.22, 0.43, 0.56),
        )
        affine_plan = BoundaryAffinePlan(
            (
                BoundaryAffineDecision(
                    0,
                    0,
                    1,
                    0.966,
                    1.0,
                    "silhouette",
                    rejected_match,
                    "insufficient_ratio_matches",
                    exact_pair,
                ),
            )
        )

        plan = build_editorial_motion_plan(
            visual_timeline=timeline,
            transitions=(RenderTransition(tier="C", reason="hard"),),
            boundary_affine_plan=affine_plan,
            config=EditorialMotionConfig(
                target_zoom_events_per_minute=0.0,
                target_roll_events_per_minute=0.0,
            ),
        )

        self.assertEqual(plan.boundary_match_count, 1)
        incoming = plan.track_for(1)
        assert incoming is not None
        self.assertGreater(incoming.keyframes[0].zoom, 1.0)
        self.assertIn(
            "exact_frame_silhouette_scale_position_matched_hard_cut",
            incoming.reason,
        )

    def test_section_plan_is_consumed_and_reported(self) -> None:
        timeline = VisualTimelinePlan(
            choices=(
                _choice(
                    0,
                    source_index=0,
                    start=0.0,
                    end=8.0,
                    shot_id="continuous",
                    face_height=0.10,
                ),
            ),
            optimizer_score=1.0,
        )
        section_plan = SectionPlan(
            (
                SectionDirective(
                    section_index=0,
                    role=SectionRole.CHORUS_LIKE,
                    start_seconds=0.0,
                    end_seconds=8.0,
                    energy_score=0.9,
                    confidence=0.8,
                    phrase_count=1,
                    preferred_source_hold_beats=2,
                    target_source_switches_per_minute=17.0,
                    camera_motion_budget=0.82,
                    target_roll_events_per_minute=0.35,
                    establish_with_wide_shot=True,
                    protect_last_beat_from_extra_cut=False,
                ),
            )
        )
        plan = build_editorial_motion_plan(
            visual_timeline=timeline,
            transitions=(),
            section_plan=section_plan,
        )
        self.assertTrue(plan.section_directed)
        self.assertTrue(plan.to_dict()["section_directed"])
        self.assertIn("section_budget", plan.to_dict()["policy"]["roll"])

    def test_cross_source_face_heights_are_matched_by_boundary_local_zoom(self) -> None:
        timeline = VisualTimelinePlan(
            choices=(
                _choice(
                    0,
                    source_index=0,
                    start=0.0,
                    end=1.0,
                    shot_id="left",
                    face_height=0.10,
                ),
                _choice(
                    1,
                    source_index=1,
                    start=1.0,
                    end=2.0,
                    shot_id="right",
                    face_height=0.11,
                ),
            ),
            optimizer_score=1.0,
        )

        plan = build_editorial_motion_plan(
            visual_timeline=timeline,
            transitions=(RenderTransition(tier="C", reason="hard"),),
        )

        self.assertEqual(plan.boundary_match_count, 1)
        left = plan.track_for(0)
        right = plan.track_for(1)
        assert left is not None
        left_zoom = left.final_keyframe.zoom
        right_zoom = right.keyframes[0].zoom if right is not None else 1.0
        self.assertAlmostEqual(0.10 * left_zoom, 0.11 * right_zoom, places=6)
        self.assertIn("face_scale_position_matched_hard_cut", left.reason)

    def test_tier_d_same_shot_run_slices_one_curve_without_zoom_reset(self) -> None:
        timeline = VisualTimelinePlan(
            choices=(
                _choice(
                    0,
                    source_index=0,
                    start=0.0,
                    end=3.5,
                    shot_id="same",
                    face_height=0.10,
                ),
                _choice(
                    1,
                    source_index=0,
                    start=3.5,
                    end=4.0,
                    shot_id="same",
                    face_height=0.10,
                ),
            ),
            optimizer_score=1.0,
        )
        plan = build_editorial_motion_plan(
            visual_timeline=timeline,
            transitions=(RenderTransition(tier="D", reason="hold"),),
            phrase_times=(4.0,),
            config=EditorialMotionConfig(target_zoom_events_per_minute=30.0),
        )

        left = plan.track_for(0)
        right = plan.track_for(1)
        assert left is not None and right is not None
        left_a, left_b = left.keyframes[-2:]
        right_a, right_b = right.keyframes[:2]
        cross_frame_slope = (
            right.keyframes[0].zoom - left.final_keyframe.zoom
        ) * 29.97
        self.assertAlmostEqual(
            _slope(
                left_a.time_seconds,
                left_a.zoom,
                left_b.time_seconds,
                left_b.zoom,
            ),
            _slope(
                right_a.time_seconds,
                right_a.zoom,
                right_b.time_seconds,
                right_b.zoom,
            ),
            places=9,
        )
        self.assertAlmostEqual(
            cross_frame_slope,
            _slope(
                right_a.time_seconds,
                right_a.zoom,
                right_b.time_seconds,
                right_b.zoom,
            ),
            places=9,
        )

    def test_dissolve_adjacent_segments_never_receive_editorial_transform(self) -> None:
        timeline = VisualTimelinePlan(
            choices=(
                _choice(
                    0,
                    source_index=0,
                    start=0.0,
                    end=1.0,
                    shot_id="left",
                    face_height=0.10,
                ),
                _choice(
                    1,
                    source_index=1,
                    start=1.0,
                    end=2.0,
                    shot_id="right",
                    face_height=0.11,
                ),
            ),
            optimizer_score=1.0,
        )
        plan = build_editorial_motion_plan(
            visual_timeline=timeline,
            transitions=(
                RenderTransition(
                    TransitionKind.FACE_DISSOLVE,
                    11,
                    "reviewed",
                    tier="A",
                ),
            ),
        )

        self.assertEqual(plan.blocked_dissolve_segment_count, 2)
        self.assertIsNone(plan.track_for(0))
        self.assertIsNone(plan.track_for(1))
        self.assertEqual(plan.boundary_match_count, 0)

    def test_head_roll_rule_is_sparse_evidence_driven_not_random(self) -> None:
        choices = (
            _choice(0, source_index=0, start=0.0, end=10.0, shot_id="first"),
            _choice(
                1,
                source_index=1,
                start=10.0,
                end=20.0,
                shot_id="second",
                face_roll_degrees=-2.0,
            ),
        )
        plan = build_editorial_motion_plan(
            visual_timeline=VisualTimelinePlan(choices=choices, optimizer_score=1.0),
            transitions=(RenderTransition(tier="C", reason="hard"),),
            config=EditorialMotionConfig(
                target_zoom_events_per_minute=0.0,
                target_roll_events_per_minute=6.0,
            ),
        )

        self.assertEqual(plan.roll_event_count, 0)
        self.assertEqual(dict(plan.authored_event_counts)["roll"], 1)
        incoming = plan.track_for(1)
        assert incoming is not None
        self.assertLess(incoming.keyframes[0].roll_degrees, 0.0)
        self.assertEqual(incoming.keyframes[-1].roll_degrees, 0.0)

    def test_applied_face_crops_are_remeasured_at_the_actual_boundary(self) -> None:
        choices = (
            _choice(
                0,
                source_index=0,
                start=0.0,
                end=1.0,
                shot_id="left",
                face_height=0.10,
            ),
            _choice(
                1,
                source_index=1,
                start=1.0,
                end=2.0,
                shot_id="right",
                face_height=0.11,
            ),
        )
        matches = FaceMatchPlan(
            (
                _face_match(0, 0, crop_height=0.80),
                _face_match(1, 1, crop_height=0.95),
            ),
            0.12,
            1.35,
            (0.5, 0.42),
        )

        plan = build_editorial_motion_plan(
            visual_timeline=VisualTimelinePlan(
                choices=choices,
                optimizer_score=1.0,
            ),
            transitions=(RenderTransition(tier="B", reason="base crops"),),
            face_match_plan=matches,
            config=EditorialMotionConfig(
                target_zoom_events_per_minute=0.0,
                target_roll_events_per_minute=0.0,
            ),
        )

        self.assertEqual(plan.boundary_match_count, 1)
        right = plan.track_for(1)
        assert right is not None
        left_output_height = 0.10 / 0.80
        right_output_height = 0.11 / 0.95 * right.keyframes[0].zoom
        self.assertAlmostEqual(left_output_height, right_output_height, places=6)
        self.assertNotIn(
            ("base_face_or_silhouette_crop_already_matches", 1),
            plan.skipped_boundary_reasons,
        )

    def test_internal_camera_cut_blocks_boundary_editorial_ramp(self) -> None:
        left = replace(
            _choice(
                0,
                source_index=0,
                start=0.0,
                end=1.0,
                shot_id="left",
                face_height=0.10,
            ),
            internal_scene_cut_master_seconds=(0.5,),
        )
        right = _choice(
            1,
            source_index=1,
            start=1.0,
            end=2.0,
            shot_id="right",
            face_height=0.11,
        )

        plan = build_editorial_motion_plan(
            visual_timeline=VisualTimelinePlan(
                choices=(left, right),
                optimizer_score=1.0,
            ),
            transitions=(RenderTransition(tier="C", reason="internal camera cut"),),
            config=EditorialMotionConfig(
                target_zoom_events_per_minute=0.0,
                target_roll_events_per_minute=0.0,
            ),
        )

        self.assertEqual(plan.boundary_match_count, 0)
        self.assertIsNone(plan.track_for(0))
        self.assertIsNone(plan.track_for(1))
        self.assertIn(("internal_camera_cut", 1), plan.skipped_boundary_reasons)

    def test_boundary_transform_must_improve_layout_not_merely_pass_threshold(self) -> None:
        left = _choice(
            0,
            source_index=0,
            start=0.0,
            end=1.0,
            shot_id="left",
            face_height=0.10,
        )
        right = _choice(
            1,
            source_index=1,
            start=1.0,
            end=2.0,
            shot_id="right",
            face_height=0.10,
        )
        left_box = (0.11, 0.37, 0.08, 0.10)
        right_box = (0.21, 0.37, 0.08, 0.10)
        left = replace(left, face_samples_xywh=(left_box,) * 3)
        right = replace(right, face_samples_xywh=(right_box,) * 3)

        plan = build_editorial_motion_plan(
            visual_timeline=VisualTimelinePlan(
                choices=(left, right),
                optimizer_score=1.0,
            ),
            transitions=(RenderTransition(tier="C", reason="hard"),),
            config=EditorialMotionConfig(maximum_center_shift=0.0),
        )

        self.assertEqual(plan.boundary_match_count, 0)
        self.assertIsNone(plan.track_for(0))
        self.assertIsNone(plan.track_for(1))
        self.assertIn(
            ("post_transform_no_layout_improvement", 1),
            plan.skipped_boundary_reasons,
        )

    def test_missing_subject_and_scale_evidence_never_fills_zoom_quota(self) -> None:
        choices = (
            _choice(0, source_index=0, start=0.0, end=2.0, shot_id="left"),
            _choice(1, source_index=1, start=2.0, end=4.0, shot_id="right"),
        )
        choices = tuple(
            replace(item, entry_shot_scale=None, exit_shot_scale=None)
            for item in choices
        )
        plan = build_editorial_motion_plan(
            visual_timeline=VisualTimelinePlan(
                choices=choices,
                optimizer_score=1.0,
            ),
            transitions=(RenderTransition(tier="C", reason="hard"),),
            config=EditorialMotionConfig(
                target_zoom_events_per_minute=30.0,
                minimum_zoom_event_spacing_seconds=0.0,
            ),
        )

        self.assertEqual(plan.zoom_event_count, 0)
        self.assertTrue(all(plan.track_for(index) is None for index in range(2)))

    def test_boundary_window_shorter_than_reference_ramp_stays_neutral(self) -> None:
        choices = (
            _choice(
                0,
                source_index=0,
                start=0.0,
                end=0.10,
                shot_id="short",
                face_height=0.10,
            ),
            _choice(
                1,
                source_index=1,
                start=0.10,
                end=1.10,
                shot_id="long",
                face_height=0.11,
            ),
        )
        plan = build_editorial_motion_plan(
            visual_timeline=VisualTimelinePlan(
                choices=choices,
                optimizer_score=1.0,
            ),
            transitions=(RenderTransition(tier="C", reason="hard"),),
        )

        self.assertEqual(plan.boundary_match_count, 0)
        self.assertIsNone(plan.track_for(0))
        self.assertIn(
            ("outgoing_window_shorter_than_reference_ramp", 1),
            plan.skipped_boundary_reasons,
        )

    def test_two_boundary_states_on_short_run_interpolate_without_neutral_punch(self) -> None:
        choices = (
            _choice(
                0,
                source_index=0,
                start=0.0,
                end=2.0,
                shot_id="left",
                face_height=0.11,
            ),
            _choice(
                1,
                source_index=1,
                start=2.0,
                end=2.8,
                shot_id="middle",
                face_height=0.10,
            ),
            _choice(
                2,
                source_index=2,
                start=2.8,
                end=4.8,
                shot_id="right",
                face_height=0.11,
            ),
        )
        plan = build_editorial_motion_plan(
            visual_timeline=VisualTimelinePlan(
                choices=choices,
                optimizer_score=1.0,
            ),
            transitions=(
                RenderTransition(tier="C", reason="hard"),
                RenderTransition(tier="C", reason="hard"),
            ),
            config=EditorialMotionConfig(
                target_zoom_events_per_minute=30.0,
                minimum_zoom_event_spacing_seconds=0.0,
            ),
        )

        self.assertEqual(plan.boundary_match_count, 2)
        middle = plan.track_for(1)
        assert middle is not None
        self.assertTrue(all(keyframe.zoom > 1.09 for keyframe in middle.keyframes))

    def test_interior_tier_d_pulse_keeps_run_boundaries_neutral_and_slices_one_curve(self) -> None:
        choices = (
            _choice(
                0,
                source_index=1,
                start=0.0,
                end=0.3,
                shot_id="lead",
                face_height=0.10,
            ),
            _choice(
                1,
                source_index=0,
                start=0.3,
                end=1.7,
                shot_id="continuous",
                face_height=0.10,
            ),
            _choice(
                2,
                source_index=0,
                start=1.7,
                end=3.5,
                shot_id="continuous",
                face_height=0.10,
            ),
            _choice(
                3,
                source_index=2,
                start=3.5,
                end=3.8,
                shot_id="tail",
                face_height=0.10,
            ),
        )
        plan = build_editorial_motion_plan(
            visual_timeline=VisualTimelinePlan(choices=choices, optimizer_score=1.0),
            transitions=(
                RenderTransition(tier="C", reason="lead cut"),
                RenderTransition(tier="D", reason="same source hold"),
                RenderTransition(tier="C", reason="tail cut"),
            ),
            phrase_times=(2.2,),
            config=EditorialMotionConfig(
                target_zoom_events_per_minute=30.0,
                minimum_zoom_event_spacing_seconds=0.0,
            ),
        )

        self.assertEqual(plan.boundary_match_count, 0)
        self.assertEqual(plan.zoom_event_count, 2)
        self.assertEqual(dict(plan.authored_event_counts)["zoom"], 1)
        self.assertEqual(plan.interior_zoom_pulse_count, 1)
        left = plan.track_for(1)
        right = plan.track_for(2)
        assert left is not None and right is not None
        self.assertEqual(left.keyframes[0].zoom, 1.0)
        self.assertEqual(right.final_keyframe.zoom, 1.0)
        self.assertTrue(
            all(
                keyframe.center_x == keyframe.center_y == 0.5
                and keyframe.roll_degrees == 0.0
                for track in (left, right)
                for keyframe in track.keyframes
            )
        )
        left_a, left_b = left.keyframes[-2:]
        right_a, right_b = right.keyframes[:2]
        cross_frame_slope = (
            right.keyframes[0].zoom - left.final_keyframe.zoom
        ) * 29.97
        self.assertAlmostEqual(
            _slope(
                left_a.time_seconds,
                left_a.zoom,
                left_b.time_seconds,
                left_b.zoom,
            ),
            _slope(
                right_a.time_seconds,
                right_a.zoom,
                right_b.time_seconds,
                right_b.zoom,
            ),
            places=9,
        )
        self.assertAlmostEqual(
            cross_frame_slope,
            _slope(
                right_a.time_seconds,
                right_a.zoom,
                right_b.time_seconds,
                right_b.zoom,
            ),
            places=9,
        )
        self.assertIn(
            "reference_calibrated_same_source_interior_zoom_pulse",
            left.reason,
        )

    def test_interior_tier_d_pulse_rejects_subject_containment_loss(self) -> None:
        safe_choices = (
            _choice(0, source_index=1, start=0.0, end=0.3, shot_id="lead", face_height=0.10),
            _choice(1, source_index=0, start=0.3, end=1.7, shot_id="continuous", face_height=0.10),
            _choice(2, source_index=0, start=1.7, end=3.5, shot_id="continuous", face_height=0.10),
            _choice(3, source_index=2, start=3.5, end=3.8, shot_id="tail", face_height=0.10),
        )
        edge_face = (0.0, 0.35, 0.08, 0.10)
        choices = tuple(
            replace(choice, face_samples_xywh=(edge_face,) * 3)
            if choice.source_index == 0
            else choice
            for choice in safe_choices
        )
        plan = build_editorial_motion_plan(
            visual_timeline=VisualTimelinePlan(choices=choices, optimizer_score=1.0),
            transitions=(
                RenderTransition(tier="C", reason="lead cut"),
                RenderTransition(tier="D", reason="same source hold"),
                RenderTransition(tier="C", reason="tail cut"),
            ),
            phrase_times=(2.2,),
            config=EditorialMotionConfig(
                target_zoom_events_per_minute=30.0,
                minimum_zoom_event_spacing_seconds=0.0,
            ),
        )

        self.assertEqual(plan.zoom_event_count, 0)
        self.assertEqual(plan.interior_zoom_pulse_count, 0)
        self.assertIsNone(plan.track_for(1))
        self.assertIsNone(plan.track_for(2))
        self.assertIn(
            ("interior_pulse_subject_would_be_clipped", 1),
            plan.skipped_boundary_reasons,
        )

    def test_interior_tier_d_pulses_obey_density_and_global_spacing_caps(self) -> None:
        choices = tuple(
            replace(
                _choice(
                    index,
                    source_index=index // 2,
                    start=index * 1.5,
                    end=(index + 1) * 1.5,
                    shot_id=f"run-{index // 2}",
                    face_height=0.10,
                ),
                entry_shot_scale=None,
                exit_shot_scale=None,
            )
            for index in range(6)
        )
        transitions = tuple(
            RenderTransition(
                tier="D" if index in (0, 2, 4) else "C",
                reason="same shot" if index in (0, 2, 4) else "source cut",
            )
            for index in range(5)
        )

        density_plan = build_editorial_motion_plan(
            visual_timeline=VisualTimelinePlan(choices=choices, optimizer_score=1.0),
            transitions=transitions,
            config=EditorialMotionConfig(
                target_zoom_events_per_minute=15.0,
                minimum_zoom_event_spacing_seconds=0.0,
            ),
        )
        self.assertEqual(density_plan.zoom_event_count, 2)
        self.assertEqual(density_plan.interior_zoom_pulse_count, 2)
        self.assertIn(
            ("interior_pulse_density_cap", 1),
            density_plan.skipped_boundary_reasons,
        )

        spacing_plan = build_editorial_motion_plan(
            visual_timeline=VisualTimelinePlan(choices=choices, optimizer_score=1.0),
            transitions=transitions,
            config=EditorialMotionConfig(
                target_zoom_events_per_minute=15.0,
                minimum_zoom_event_spacing_seconds=3.5,
            ),
        )
        self.assertEqual(spacing_plan.zoom_event_count, 2)
        self.assertEqual(spacing_plan.interior_zoom_pulse_count, 2)
        self.assertIn(
            ("interior_pulse_spacing_cap", 1),
            spacing_plan.skipped_boundary_reasons,
        )
        self.assertTrue(spacing_plan.track_for(0) is not None)
        self.assertTrue(spacing_plan.track_for(1) is not None)
        self.assertIsNone(spacing_plan.track_for(2))
        self.assertIsNone(spacing_plan.track_for(3))
        self.assertTrue(spacing_plan.track_for(4) is not None)
        self.assertTrue(spacing_plan.track_for(5) is not None)

    def test_interior_tier_d_pulse_has_no_effect_on_short_or_unsafe_runs(self) -> None:
        short_choices = tuple(
            replace(
                _choice(
                    index,
                    source_index=0,
                    start=index * 1.2,
                    end=(index + 1) * 1.2,
                    shot_id="short",
                    face_height=0.10,
                ),
                entry_shot_scale=None,
                exit_shot_scale=None,
            )
            for index in range(2)
        )
        short_plan = build_editorial_motion_plan(
            visual_timeline=VisualTimelinePlan(
                choices=short_choices,
                optimizer_score=1.0,
            ),
            transitions=(RenderTransition(tier="D", reason="same shot"),),
            config=EditorialMotionConfig(
                target_zoom_events_per_minute=30.0,
                minimum_zoom_event_spacing_seconds=0.0,
            ),
        )
        self.assertEqual(short_plan.zoom_event_count, 0)
        self.assertTrue(all(short_plan.track_for(index) is None for index in range(2)))
        self.assertIn(
            ("interior_pulse_run_too_short_for_ramps_and_guards", 1),
            short_plan.skipped_boundary_reasons,
        )

        unsafe_choices = tuple(
            replace(
                _choice(
                    index,
                    source_index=0,
                    start=index * 1.5,
                    end=(index + 1) * 1.5,
                    shot_id="unsafe",
                    face_height=0.10,
                ),
                entry_shot_scale=None,
                exit_shot_scale=None,
                internal_scene_cut_master_seconds=((0.8,) if index == 0 else ()),
            )
            for index in range(2)
        )
        unsafe_plan = build_editorial_motion_plan(
            visual_timeline=VisualTimelinePlan(
                choices=unsafe_choices,
                optimizer_score=1.0,
            ),
            transitions=(RenderTransition(tier="D", reason="same shot"),),
            config=EditorialMotionConfig(
                target_zoom_events_per_minute=30.0,
                minimum_zoom_event_spacing_seconds=0.0,
            ),
        )
        self.assertEqual(unsafe_plan.zoom_event_count, 0)
        self.assertTrue(all(unsafe_plan.track_for(index) is None for index in range(2)))
        self.assertIn(
            ("interior_pulse_internal_or_native_cut", 1),
            unsafe_plan.skipped_boundary_reasons,
        )

        crop_choices = tuple(
            replace(choice, internal_scene_cut_master_seconds=())
            for choice in unsafe_choices
        )
        crop_plan = build_editorial_motion_plan(
            visual_timeline=VisualTimelinePlan(
                choices=crop_choices,
                optimizer_score=1.0,
            ),
            transitions=(RenderTransition(tier="D", reason="same shot"),),
            face_match_plan=FaceMatchPlan(
                (_face_match(0, 0, crop_height=0.90),),
                0.12,
                1.35,
                (0.5, 0.42),
            ),
            config=EditorialMotionConfig(
                target_zoom_events_per_minute=30.0,
                minimum_zoom_event_spacing_seconds=0.0,
            ),
        )
        self.assertEqual(crop_plan.zoom_event_count, 0)
        self.assertTrue(all(crop_plan.track_for(index) is None for index in range(2)))
        self.assertIn(
            ("interior_pulse_applied_face_crop", 1),
            crop_plan.skipped_boundary_reasons,
        )

    def test_fractional_global_breakpoint_is_frame_quantized_before_tier_d_slice(self) -> None:
        fps = 29.97
        first_duration = 30.0 / fps
        second_duration = 23.0 / fps
        choices = (
            _choice(
                0,
                source_index=0,
                start=0.0,
                end=first_duration,
                shot_id="continuous",
                face_height=0.10,
            ),
            _choice(
                1,
                source_index=0,
                start=first_duration,
                end=first_duration + second_duration,
                shot_id="continuous",
                face_height=0.10,
            ),
        )
        plan = build_editorial_motion_plan(
            visual_timeline=VisualTimelinePlan(choices=choices, optimizer_score=1.0),
            transitions=(RenderTransition(tier="D", reason="same shot"),),
            fps=fps,
            config=EditorialMotionConfig(
                target_zoom_events_per_minute=30.0,
                minimum_zoom_event_spacing_seconds=0.0,
            ),
        )

        raw_breakpoint = 52.0 / fps - 0.75
        self.assertLess(29.0 / fps, raw_breakpoint)
        self.assertLess(raw_breakpoint, 30.0 / fps)
        self.assertIsNone(plan.track_for(0))
        incoming = plan.track_for(1)
        assert incoming is not None
        self.assertEqual(incoming.keyframes[0].zoom, 1.0)

        render_plan = RenderPlan(
            sources=(Path("one.mp4"),),
            segments=tuple(
                RenderSegment(
                    source_index=choice.source_index,
                    source_start=choice.source_local_start,
                    duration=choice.duration_seconds,
                    master_start=choice.canonical_master_start,
                    editorial_transform=plan.track_for(index),
                )
                for index, choice in enumerate(choices)
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=fps,
            transitions=(RenderTransition(tier="D", reason="same shot"),),
        )
        self.assertEqual(len(render_plan.segments), 2)


if __name__ == "__main__":
    unittest.main()
