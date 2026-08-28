from __future__ import annotations

import unittest

import numpy as np

from crosscut.ive import EditMode
from crosscut.pose import FormationPersonDescriptor
from crosscut.timeline import (
    BoundarySideStability,
    ShotType,
    TimelineCandidate,
    TimelineConfig,
    TimelineMode,
    cross_source_layout_mismatch,
    face_geometry_minimax_mismatch,
    formation_assignment_mismatch,
    select_timeline,
    select_timeline_candidates,
)
from crosscut.vision import (
    CameraFallback,
    IVEMember,
    SceneDetectionConfig,
    TrackObservation,
    VirtualCameraConfig,
    detect_scene_changes,
    ive_member_stable_id,
    plan_virtual_camera,
)


def candidate(
    segment: int,
    source: str,
    quality: float,
    **kwargs: object,
) -> TimelineCandidate:
    return TimelineCandidate(
        segment_index=segment,
        source_id=source,
        start_seconds=float(segment),
        end_seconds=float(segment + 1),
        quality_score=quality,
        **kwargs,
    )


def face_geometry(
    center_x: float = 0.5,
    center_y: float = 0.4,
    eye_scale: float = 0.12,
    roll: float = 0.0,
) -> tuple[float, ...]:
    half_screen_span = eye_scale * 0.45
    return (
        center_x - half_screen_span,
        center_y,
        center_x + half_screen_span,
        center_y,
        eye_scale,
        roll,
        0.0,
        0.75,
        0.0,
        1.45,
    )


def continuity_gate_config(**overrides: object) -> TimelineConfig:
    """Isolate one transition signal while exercising the hard-gate path."""

    values: dict[str, object] = {
        "quality_weight": 1.0,
        "sync_weight": 0.0,
        "composition_weight": 0.0,
        "boundary_weight": 0.0,
        "continuity_bonus": 0.0,
        "cut_penalty": 0.0,
        "early_cut_penalty": 0.0,
        "long_hold_penalty": 0.0,
        "pose_transition_weight": 0.0,
        "silhouette_transition_weight": 0.0,
        "motion_transition_weight": 0.0,
        "scale_transition_weight": 0.0,
        "color_transition_weight": 0.0,
        "face_geometry_transition_weight": 0.0,
        "focus_position_transition_weight": 0.0,
        "subject_geometry_transition_weight": 0.0,
        "person_count_transition_weight": 0.0,
        "person_layout_transition_weight": 0.0,
        "formation_transition_weight": 0.0,
        "shot_type_transition_weight": 0.0,
        "face_identity_transition_weight": 0.0,
        "minimum_hold_segments": 1,
    }
    values.update(overrides)
    return TimelineConfig(**values)


def formation_person(
    center_x: float,
    *,
    raised_arms: bool = False,
) -> FormationPersonDescriptor:
    wrist_y = -0.95 if raised_arms else 0.05
    points = (
        (-0.24, -0.36, 0.9),
        (0.24, -0.36, 0.9),
        (-0.38, -0.10, 0.9),
        (0.38, -0.10, 0.9),
        (-0.48, wrist_y, 0.9),
        (0.48, wrist_y, 0.9),
        (-0.16, 0.38, 0.9),
        (0.16, 0.38, 0.9),
    )
    return FormationPersonDescriptor(
        bbox_xywh=(center_x - 0.09, 0.16, 0.18, 0.72),
        centroid_xy=(center_x, 0.52),
        detection_confidence=0.9,
        upper_body_keypoints_xyc=points,
    )


class SceneChangeTests(unittest.TestCase):
    def test_combined_detector_finds_hard_cuts_with_minimum_gap(self) -> None:
        black = np.zeros((40, 60, 3), dtype=np.uint8)
        white = np.full((40, 60, 3), 255, dtype=np.uint8)
        frames = [black, black, white, white, black]

        changes = detect_scene_changes(
            frames,
            config=SceneDetectionConfig(threshold=0.3, min_gap_frames=2),
        )

        self.assertEqual([change.frame_index for change in changes], [2, 4])
        self.assertTrue(all(change.score >= 0.3 for change in changes))

    def test_fast_motion_without_histogram_change_is_not_a_camera_cut(self) -> None:
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        frame[:, :30] = 220
        shifted = np.roll(frame, 25, axis=1)

        changes = detect_scene_changes(
            (frame, shifted),
            config=SceneDetectionConfig(min_gap_frames=1),
        )

        self.assertEqual(changes, ())

    def test_scene_component_gates_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimum_histogram_distance"):
            SceneDetectionConfig(minimum_histogram_distance=1.1)


class VirtualCameraTests(unittest.TestCase):
    def observation(
        self,
        frame: int,
        member: IVEMember,
        center_x: float,
        *,
        detection: float = 0.95,
        identity: float = 0.95,
    ) -> TrackObservation:
        return TrackObservation(
            frame_index=frame,
            bbox_xyxy=(center_x - 0.08, 0.2, center_x + 0.08, 0.8),
            detection_confidence=detection,
            member=member,
            member_confidence=identity,
            source_id="inkigayo",
            track_id=f"track-{frame}",
            identity_id=f"ive:{ive_member_stable_id(member)}",
        )

    def test_ive_alias_and_opaque_identity_are_preserved(self) -> None:
        observation = TrackObservation(
            frame_index=3,
            bbox_xyxy=(0.2, 0.1, 0.5, 0.9),
            detection_confidence=0.9,
            member="가을",
            member_confidence=0.8,
            identity_id="reid-cluster-17",
        )

        self.assertIs(observation.member, IVEMember.GAEUL)
        self.assertEqual(observation.identity_id, "reid-cluster-17")
        self.assertAlmostEqual(observation.confidence_for("gaeul"), 0.8)
        self.assertEqual(observation.confidence_for("rei"), 0.0)

    def test_wrong_member_and_low_identity_trigger_safe_fallback(self) -> None:
        observations = [
            self.observation(0, IVEMember.GAEUL, 0.35),
            self.observation(1, IVEMember.JANGWONYOUNG, 0.75),
            self.observation(2, IVEMember.GAEUL, 0.8, identity=0.3),
        ]
        frames = plan_virtual_camera(
            observations,
            target_member="가을",
            config=VirtualCameraConfig(lookahead_frames=0, hold_frames=1),
        )

        self.assertTrue(frames[0].safe_for_target_crop)
        self.assertEqual(frames[1].fallback, CameraFallback.HOLD_LAST)
        self.assertFalse(frames[1].accepted_observation)
        self.assertEqual(frames[2].fallback, CameraFallback.GROUP_WIDE_REQUIRED)
        self.assertIsNone(frames[2].crop)

    def test_camera_center_is_deadzone_and_speed_limited(self) -> None:
        observations = [
            self.observation(0, IVEMember.REI, 0.25),
            self.observation(1, IVEMember.REI, 0.75),
        ]
        frames = plan_virtual_camera(
            observations,
            target_member=IVEMember.REI,
            config=VirtualCameraConfig(
                lookahead_frames=0,
                deadzone_x=0.0,
                deadzone_y=0.0,
                smoothing=1.0,
                max_center_speed=0.04,
            ),
        )

        assert frames[0].crop is not None and frames[1].crop is not None
        movement = frames[1].crop.center[0] - frames[0].crop.center[0]
        self.assertGreater(movement, 0.0)
        self.assertLessEqual(movement, 0.04 + 1e-9)
        self.assertFalse(frames[1].safe_for_target_crop)

    def test_deadzone_suppresses_jitter_and_lookahead_prepositions(self) -> None:
        jitter = [
            self.observation(0, IVEMember.LIZ, 0.50),
            self.observation(1, IVEMember.LIZ, 0.52),
        ]
        stable = plan_virtual_camera(
            jitter,
            target_member=IVEMember.LIZ,
            config=VirtualCameraConfig(
                lookahead_frames=0,
                deadzone_x=0.20,
                smoothing=1.0,
            ),
        )
        assert stable[0].crop is not None and stable[1].crop is not None
        self.assertAlmostEqual(stable[0].crop.center[0], stable[1].crop.center[0])

        moving = [
            self.observation(0, IVEMember.LIZ, 0.30),
            self.observation(1, IVEMember.LIZ, 0.50),
        ]
        without_lookahead = plan_virtual_camera(
            moving,
            target_member=IVEMember.LIZ,
            config=VirtualCameraConfig(lookahead_frames=0),
        )
        with_lookahead = plan_virtual_camera(
            moving,
            target_member=IVEMember.LIZ,
            config=VirtualCameraConfig(lookahead_frames=1),
        )
        assert without_lookahead[0].crop is not None
        assert with_lookahead[0].crop is not None
        self.assertGreater(
            with_lookahead[0].crop.center[0],
            without_lookahead[0].crop.center[0],
        )

    def test_normalized_crop_preserves_requested_pixel_aspect(self) -> None:
        frames = plan_virtual_camera(
            [self.observation(0, IVEMember.REI, 0.5)],
            target_member=IVEMember.REI,
            config=VirtualCameraConfig(
                output_aspect_ratio=9 / 16,
                source_aspect_ratio=16 / 9,
                lookahead_frames=0,
            ),
        )
        crop = frames[0].crop
        assert crop is not None
        pixel_aspect = (crop.width * 16) / (crop.height * 9)
        self.assertAlmostEqual(pixel_aspect, 9 / 16, places=6)


class TimelineOptimisationTests(unittest.TestCase):
    @staticmethod
    def maximum_source_run(sources: list[str]) -> int:
        maximum = run = 0
        previous: str | None = None
        for source in sources:
            run = run + 1 if source == previous else 1
            maximum = max(maximum, run)
            previous = source
        return maximum

    def test_global_optimisation_beats_greedy_first_pick(self) -> None:
        segments = [
            [candidate(0, "A", 0.90), candidate(0, "B", 0.80)],
            [candidate(1, "A", 0.20), candidate(1, "B", 0.95)],
        ]
        config = TimelineConfig(
            quality_weight=1.0,
            sync_weight=0.0,
            composition_weight=0.0,
            boundary_weight=0.0,
            continuity_bonus=0.30,
            cut_penalty=0.40,
            early_cut_penalty=0.0,
            pose_transition_weight=0.0,
            motion_transition_weight=0.0,
            scale_transition_weight=0.0,
            minimum_hold_segments=1,
        )

        result = select_timeline(segments, config=config)

        self.assertEqual([item.source_id for item in result.candidates], ["B", "B"])

    def test_quality_soft_floor_discourages_genuinely_degraded_shots(self) -> None:
        segments = [[
            candidate(0, "soft", 0.20, composition_score=1.0),
            candidate(0, "clean", 0.43, composition_score=0.30),
        ]]
        base = continuity_gate_config(composition_weight=0.40)
        protected = continuity_gate_config(
            composition_weight=0.40,
            quality_soft_floor=0.42,
            quality_soft_floor_penalty=0.65,
        )

        self.assertEqual(
            select_timeline(segments, config=base).candidates[0].source_id,
            "soft",
        )
        protected_result = select_timeline(segments, config=protected)
        self.assertEqual(protected_result.candidates[0].source_id, "clean")
        self.assertAlmostEqual(
            protected_result.choices[0].emission_score,
            0.43 + 0.40 * 0.30,
        )

    def test_quality_soft_floor_does_not_change_above_floor_emission(self) -> None:
        segments = [[candidate(0, "clean", 0.75, composition_score=0.40)]]
        base = continuity_gate_config(composition_weight=0.30)
        protected = continuity_gate_config(
            composition_weight=0.30,
            quality_soft_floor=0.42,
            quality_soft_floor_penalty=0.65,
        )

        base_score = select_timeline(segments, config=base).choices[0].emission_score
        protected_score = select_timeline(
            segments,
            config=protected,
        ).choices[0].emission_score

        self.assertEqual(protected_score, base_score)

    def test_quality_soft_floor_configuration_is_fail_closed(self) -> None:
        for value in (-0.01, 1.01, float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "quality_soft_floor"):
                    TimelineConfig(quality_soft_floor=value)
        with self.assertRaisesRegex(ValueError, "quality_soft_floor_penalty"):
            TimelineConfig(quality_soft_floor=0.42, quality_soft_floor_penalty=-0.01)
        with self.assertRaisesRegex(ValueError, "requires quality_soft_floor"):
            TimelineConfig(quality_soft_floor_penalty=0.65)

    def test_product_edit_mode_is_accepted(self) -> None:
        result = select_timeline(
            [[candidate(0, "A", 0.8)]], mode=EditMode.GROUP
        )

        self.assertEqual(result.mode, TimelineMode.GENERAL)

    def test_internal_camera_cut_keeps_continuity_without_cut_penalty(self) -> None:
        segments = [
            [candidate(0, "A", 0.8, shot_id="A:shot-1")],
            [
                candidate(1, "A", 0.8, shot_id="A:shot-2"),
                candidate(1, "B", 0.8, shot_id="B:shot-1"),
            ],
        ]
        config = TimelineConfig(
            quality_weight=0.0,
            sync_weight=0.0,
            composition_weight=0.0,
            boundary_weight=0.0,
            continuity_bonus=1.0,
            cut_penalty=0.2,
            early_cut_penalty=0.0,
            pose_transition_weight=0.0,
            motion_transition_weight=0.0,
            scale_transition_weight=0.0,
            minimum_hold_segments=1,
        )

        result = select_timeline(segments, config=config)

        self.assertEqual(result.choices[1].transition_score, 1.0)

    def test_native_camera_cuts_do_not_drive_unnecessary_source_switches(
        self,
    ) -> None:
        segments = []
        for index in range(8):
            native_shot = index // 2
            options = [
                candidate(
                    index,
                    "A",
                    0.92,
                    shot_id=f"A:shot-{native_shot}",
                    # Deliberately extreme descriptor changes prove that a
                    # detected same-source broadcaster cut skips visual
                    # mismatch penalties as well as the generic cut penalty.
                    entry_shot_scale=0.1 if native_shot % 2 == 0 else 0.9,
                    exit_shot_scale=0.1 if native_shot % 2 == 0 else 0.9,
                    entry_pose_embedding=(float(native_shot),) * 2,
                    exit_pose_embedding=(float(native_shot),) * 2,
                ),
                candidate(index, "B", 0.90, shot_id="B:continuous"),
            ]
            if index:
                options.append(
                    candidate(
                        index,
                        "C",
                        1.0,
                        shot_id="C:unsafe-high",
                        entry_scene_cut_near=True,
                    )
                )
            segments.append(options)

        result = select_timeline(
            segments,
            config=continuity_gate_config(
                continuity_bonus=0.05,
                cut_penalty=0.20,
                pose_transition_weight=1.0,
                scale_transition_weight=1.0,
            ),
        )

        selected = list(result.candidates)
        sources = [item.source_id for item in selected]
        native_cuts = sum(
            left.source_id == right.source_id and left.shot_id != right.shot_id
            for left, right in zip(selected, selected[1:])
        )
        source_switches = sum(
            left.source_id != right.source_id
            for left, right in zip(selected, selected[1:])
        )
        self.assertEqual(sources, ["A"] * 8)
        self.assertEqual(native_cuts, 3)
        self.assertEqual(source_switches, 0)
        self.assertNotIn("C", sources)
        self.assertTrue(
            all(
                choice.transition_score == 0.05
                for choice in result.choices[1:]
            )
        )

    def test_source_switch_bonus_increases_safe_camera_changes(self) -> None:
        segments = [
            [
                candidate(index, "A", 1.0),
                candidate(index, "B", 0.96),
            ]
            for index in range(4)
        ]

        without_bonus = select_timeline(
            segments,
            config=continuity_gate_config(source_switch_bonus=0.0),
        )
        with_bonus = select_timeline(
            segments,
            config=continuity_gate_config(source_switch_bonus=0.06),
        )

        without_sources = [item.source_id for item in without_bonus.candidates]
        with_sources = [item.source_id for item in with_bonus.candidates]
        without_switches = sum(
            previous != current
            for previous, current in zip(without_sources, without_sources[1:])
        )
        with_switches = sum(
            previous != current
            for previous, current in zip(with_sources, with_sources[1:])
        )

        self.assertEqual(without_switches, 0)
        self.assertEqual(with_switches, 3)
        self.assertGreater(with_switches, without_switches)
        self.assertTrue(
            all(
                abs(choice.transition_score - 0.06) < 1e-12
                for choice in with_bonus.choices[1:]
            )
        )

    def test_source_switch_bonus_must_be_nonnegative(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_switch_bonus"):
            TimelineConfig(source_switch_bonus=-0.01)

    def test_adaptive_source_cap_rotates_a_nine_segment_edit(self) -> None:
        segments = [
            [
                candidate(index, "A", 0.82, shot_id=f"A:shot-{index}"),
                candidate(index, "B", 0.80, shot_id="B:continuous"),
            ]
            for index in range(9)
        ]
        config = TimelineConfig(
            quality_weight=1.0,
            sync_weight=0.0,
            composition_weight=0.0,
            boundary_weight=0.0,
            continuity_bonus=0.0,
            cut_penalty=0.0,
            early_cut_penalty=0.0,
            long_hold_penalty=0.0,
            source_hold_penalty=0.2,
            pose_transition_weight=0.0,
            motion_transition_weight=0.0,
            scale_transition_weight=0.0,
            minimum_hold_segments=1,
            preferred_max_source_hold_segments=3,
            maximum_same_source_hold_segments=4,
            maximum_switch_score_sacrifice=0.1,
        )

        result = select_timeline(segments, config=config)
        sources = [item.source_id for item in result.candidates]

        self.assertEqual(len(sources), 9)
        self.assertEqual(set(sources), {"A", "B"})
        self.assertLessEqual(self.maximum_source_run(sources), 4)

    def test_source_cap_does_not_force_a_bad_alternate(self) -> None:
        segments = [
            [candidate(index, "good", 1.0), candidate(index, "bad", 0.05)]
            for index in range(9)
        ]
        config = TimelineConfig(
            quality_weight=1.0,
            sync_weight=0.0,
            composition_weight=0.0,
            boundary_weight=0.0,
            continuity_bonus=0.0,
            cut_penalty=0.0,
            early_cut_penalty=0.0,
            long_hold_penalty=0.0,
            source_hold_penalty=1.0,
            pose_transition_weight=0.0,
            motion_transition_weight=0.0,
            scale_transition_weight=0.0,
            minimum_hold_segments=1,
            preferred_max_source_hold_segments=2,
            maximum_same_source_hold_segments=3,
            maximum_switch_score_sacrifice=0.2,
        )

        result = select_timeline(segments, config=config)

        self.assertEqual(
            [item.source_id for item in result.candidates],
            ["good"] * 9,
        )

    def test_time_based_source_cap_handles_unequal_segment_lengths(self) -> None:
        spans = ((0.0, 1.0), (1.0, 2.0), (2.0, 6.5), (6.5, 7.5))
        segments = []
        for index, (start, end) in enumerate(spans):
            segments.append(
                [
                    TimelineCandidate(
                        segment_index=index,
                        source_id="A",
                        start_seconds=start,
                        end_seconds=end,
                        quality_score=1.0,
                    ),
                    TimelineCandidate(
                        segment_index=index,
                        source_id="B",
                        start_seconds=start,
                        end_seconds=end,
                        quality_score=0.98,
                    ),
                ]
            )

        result = select_timeline(
            segments,
            config=continuity_gate_config(
                maximum_same_source_hold_seconds=5.0,
                maximum_switch_score_sacrifice=0.1,
            ),
        )

        self.assertNotEqual(
            result.candidates[1].source_id,
            result.candidates[2].source_id,
        )

    def test_time_based_source_cap_is_absolute_across_internal_camera_cuts(
        self,
    ) -> None:
        spans = tuple(
            (float(index * 2), float((index + 1) * 2)) for index in range(5)
        )
        segments = [
            [
                TimelineCandidate(
                    segment_index=index,
                    source_id="A",
                    start_seconds=start,
                    end_seconds=end,
                    quality_score=1.0,
                    shot_id=f"A:shot-{index}",
                ),
                TimelineCandidate(
                    segment_index=index,
                    source_id="B",
                    start_seconds=start,
                    end_seconds=end,
                    quality_score=0.05,
                    shot_id=f"B:shot-{index}",
                ),
            ]
            for index, (start, end) in enumerate(spans)
        ]

        result = select_timeline(
            segments,
            config=continuity_gate_config(
                maximum_same_source_hold_seconds=5.0,
                maximum_switch_score_sacrifice=0.1,
            ),
        )

        source_runs: list[float] = []
        previous_source: str | None = None
        for selected in result.candidates:
            duration = selected.end_seconds - selected.start_seconds
            if selected.source_id == previous_source:
                source_runs[-1] += duration
            else:
                source_runs.append(duration)
            previous_source = selected.source_id
        self.assertEqual(
            set(item.source_id for item in result.candidates),
            {"A", "B"},
        )
        self.assertLessEqual(max(source_runs), 5.0)

    def test_time_based_source_cap_rejects_an_oversized_decision_span(self) -> None:
        segments = [
            [
                TimelineCandidate(
                    segment_index=0,
                    source_id=source,
                    start_seconds=0.0,
                    end_seconds=6.0,
                    quality_score=1.0,
                )
                for source in ("A", "B")
            ]
        ]

        with self.assertRaisesRegex(
            ValueError,
            "segment 0 is longer than the maximum same-source hold seconds",
        ):
            select_timeline(
                segments,
                config=continuity_gate_config(
                    maximum_same_source_hold_seconds=5.0,
                ),
            )

    def test_dead_end_reports_transition_and_hold_rejection_counts(self) -> None:
        spans = ((0.0, 2.0), (2.0, 4.0))
        segments = [
            [
                TimelineCandidate(
                    segment_index=index,
                    source_id=source,
                    start_seconds=start,
                    end_seconds=end,
                    quality_score=1.0,
                    shot_scale=scale,
                )
                for source, scale in (("A", 0.1), ("B", 0.9))
            ]
            for index, (start, end) in enumerate(spans)
        ]

        with self.assertRaises(ValueError) as raised:
            select_timeline(
                segments,
                config=continuity_gate_config(
                    preferred_max_source_hold_seconds=2.0,
                    maximum_same_source_hold_seconds=2.0,
                    maximum_cross_source_scale_delta=0.2,
                ),
            )

        message = str(raised.exception)
        self.assertIn("segment 1 has no cadence-safe candidate", message)
        self.assertIn("same_source_duration_hold_cap:2", message)
        self.assertIn("scale_delta:2", message)

    def test_ping_pong_penalty_avoids_immediate_source_return(self) -> None:
        qualities = (
            {"A": 1.0, "B": 0.8, "C": 0.8},
            {"A": 0.8, "B": 1.0, "C": 0.8},
            {"A": 1.0, "B": 0.8, "C": 0.98},
        )
        segments = [
            [candidate(index, source, score) for source, score in scores.items()]
            for index, scores in enumerate(qualities)
        ]

        no_history = select_timeline(
            segments,
            config=continuity_gate_config(
                source_switch_bonus=0.05,
                ping_pong_penalty=0.0,
            ),
        )
        with_history = select_timeline(
            segments,
            config=continuity_gate_config(
                source_switch_bonus=0.05,
                ping_pong_penalty=0.25,
            ),
        )

        self.assertEqual(
            [item.source_id for item in no_history.candidates],
            ["A", "B", "A"],
        )
        self.assertEqual(
            [item.source_id for item in with_history.candidates],
            ["A", "B", "C"],
        )

    def test_hold_seconds_configuration_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            TimelineConfig(preferred_max_source_hold_seconds=0.0)
        with self.assertRaisesRegex(ValueError, "below preferred seconds"):
            TimelineConfig(
                preferred_max_source_hold_seconds=5.0,
                maximum_same_source_hold_seconds=4.0,
            )

    def test_generates_distinct_globally_safe_timeline_candidates(self) -> None:
        segments = [
            [
                candidate(index, "A", 0.90),
                candidate(index, "B", 0.89),
                candidate(index, "C", 0.88),
            ]
            for index in range(12)
        ]
        results = select_timeline_candidates(
            segments,
            config=continuity_gate_config(),
            maximum_results=5,
            diversity_penalty=0.08,
            minimum_hamming_fraction=0.10,
        )
        signatures = {
            tuple(candidate.source_id for candidate in result.candidates)
            for result in results
        }
        self.assertEqual(len(results), 5)
        self.assertEqual(len(signatures), 5)
        self.assertTrue(all(len(result.candidates) == 12 for result in results))

    def test_candidate_generation_validates_search_limits(self) -> None:
        segments = [[candidate(0, "A", 1.0)]]
        with self.assertRaisesRegex(ValueError, "1 to 32"):
            select_timeline_candidates(segments, maximum_results=33)
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            select_timeline_candidates(segments, diversity_penalty=0.0)
        with self.assertRaisesRegex(ValueError, "non-positive"):
            select_timeline(
                segments,
                emission_adjustments={(0, "A"): 0.1},
            )

    def test_coverage_preferences_rematch_an_unavailable_final_slot(self) -> None:
        source_ids = tuple(str(index) for index in range(9))
        segments = [
            [
                candidate(segment, source_id, 0.8)
                for source_id in source_ids
                if not (segment == 8 and source_id == "8")
            ]
            for segment in range(9)
        ]
        config = TimelineConfig(
            quality_weight=1.0,
            sync_weight=0.0,
            composition_weight=0.0,
            boundary_weight=0.0,
            continuity_bonus=0.0,
            cut_penalty=0.0,
            early_cut_penalty=0.0,
            pose_transition_weight=0.0,
            motion_transition_weight=0.0,
            scale_transition_weight=0.0,
            minimum_hold_segments=1,
            preferred_source_bonus=1.0,
            preferred_source_max_score_sacrifice=0.1,
        )

        result = select_timeline(
            segments,
            config=config,
            preferred_source_ids=source_ids,
        )

        self.assertEqual(
            {item.source_id for item in result.candidates},
            set(source_ids),
        )

    def test_candidate_group_requires_one_canonical_span(self) -> None:
        left = candidate(0, "A", 0.8)
        right = TimelineCandidate(
            segment_index=0,
            source_id="B",
            start_seconds=0.1,
            end_seconds=1.0,
            quality_score=0.8,
        )
        with self.assertRaisesRegex(ValueError, "canonical span"):
            select_timeline([[left, right]])

    def test_cut_compares_outgoing_pose_to_incoming_pose(self) -> None:
        segments = [
            [
                candidate(
                    0,
                    "A",
                    0.5,
                    entry_pose_embedding=(0.0, 0.0),
                    exit_pose_embedding=(1.0, 1.0),
                )
            ],
            [
                candidate(
                    1,
                    "B",
                    0.5,
                    entry_pose_embedding=(1.0, 1.0),
                    exit_pose_embedding=(0.0, 0.0),
                )
            ],
        ]
        config = TimelineConfig(
            quality_weight=0.0,
            sync_weight=0.0,
            composition_weight=0.0,
            boundary_weight=0.0,
            cut_penalty=0.0,
            early_cut_penalty=0.0,
            pose_transition_weight=1.0,
            motion_transition_weight=0.0,
            scale_transition_weight=0.0,
            minimum_hold_segments=1,
        )

        result = select_timeline(segments, config=config)

        self.assertAlmostEqual(result.choices[1].transition_score, 0.0)

    def test_cross_source_scale_gate_keeps_same_source_internal_cut(self) -> None:
        segments = [
            [candidate(0, "A", 0.5, shot_scale=0.1, shot_id="A:shot-1")],
            [
                candidate(1, "A", 0.0, shot_scale=0.9, shot_id="A:shot-2"),
                candidate(1, "B", 1.0, shot_scale=0.9, shot_id="B:shot-1"),
            ],
        ]

        result = select_timeline(
            segments,
            config=continuity_gate_config(
                maximum_cross_source_scale_delta=0.2,
                source_switch_bonus=10.0,
            ),
        )

        self.assertEqual([item.source_id for item in result.candidates], ["A", "A"])
        self.assertAlmostEqual(result.choices[1].transition_score, 0.0)

    def test_scale_gate_uses_exit_and_entry_not_segment_middle(self) -> None:
        segments = [
            [
                candidate(
                    0,
                    "A",
                    0.5,
                    shot_scale=0.4,
                    exit_shot_scale=0.1,
                    shot_id="A:shot-1",
                )
            ],
            [
                candidate(
                    1,
                    "A",
                    0.0,
                    shot_scale=0.4,
                    entry_shot_scale=0.9,
                    shot_id="A:shot-2",
                ),
                candidate(
                    1,
                    "B",
                    1.0,
                    shot_scale=0.4,
                    entry_shot_scale=0.9,
                    shot_id="B:shot-1",
                ),
            ],
        ]

        result = select_timeline(
            segments,
            config=continuity_gate_config(
                maximum_cross_source_scale_delta=0.2,
            ),
        )

        self.assertEqual([item.source_id for item in result.candidates], ["A", "A"])

    def test_closeup_gate_uses_boundary_shot_types(self) -> None:
        segments = [
            [
                candidate(
                    0,
                    "A",
                    0.5,
                    shot_type=ShotType.GROUP_WIDE,
                    exit_shot_type=ShotType.CLOSE_UP,
                    shot_id="A:shot-1",
                )
            ],
            [
                candidate(
                    1,
                    "A",
                    0.0,
                    shot_type=ShotType.GROUP_WIDE,
                    entry_shot_type=ShotType.CLOSE_UP,
                    shot_id="A:shot-2",
                ),
                candidate(
                    1,
                    "B",
                    1.0,
                    shot_type=ShotType.GROUP_WIDE,
                    entry_shot_type=ShotType.CLOSE_UP,
                    shot_id="B:shot-1",
                ),
            ],
        ]

        result = select_timeline(
            segments,
            config=continuity_gate_config(
                minimum_cross_source_face_similarity=0.36,
            ),
        )

        self.assertEqual([item.source_id for item in result.candidates], ["A", "A"])

    def test_nearby_broadcast_camera_cut_blocks_cross_source_edge(self) -> None:
        visual_evidence = (0.1, 0.2, 0.3)
        segments = [
            [
                candidate(
                    0,
                    "A",
                    0.5,
                    exit_person_layout_embedding=visual_evidence,
                    exit_scene_cut_near=True,
                    shot_id="A:shot-1",
                )
            ],
            [
                candidate(
                    1,
                    "A",
                    0.0,
                    entry_person_layout_embedding=visual_evidence,
                    shot_id="A:shot-2",
                ),
                candidate(
                    1,
                    "B",
                    1.0,
                    entry_person_layout_embedding=visual_evidence,
                    shot_id="B:shot-1",
                ),
            ],
        ]

        result = select_timeline(
            segments,
            config=continuity_gate_config(
                require_cross_source_visual_evidence=True,
            ),
        )

        self.assertEqual([item.source_id for item in result.candidates], ["A", "A"])

    def test_outgoing_one_frame_camera_flash_blocks_cross_source_edge(self) -> None:
        # Production boundary 32: source 9 is group-wide until one close-up
        # frame (.163 -> .677), then source 3 begins on a compatible close-up.
        # Cross-boundary scale alone is only .092 and must not hide the double-cut.
        outgoing_flash = BoundarySideStability(
            sample_span_seconds=3.0 / 29.97,
            scale_delta=0.6768598338 - 0.1631326356,
            structure_mismatch=0.51,
            shot_type_mismatch=1.0,
        )
        stable_incoming = BoundarySideStability(
            sample_span_seconds=3.0 / 29.97,
            scale_delta=0.02,
            structure_mismatch=0.04,
            shot_type_mismatch=0.0,
        )
        segments = [
            [
                candidate(
                    0,
                    "9",
                    0.5,
                    shot_scale=0.18,
                    entry_shot_scale=0.1631326356,
                    exit_shot_scale=0.6768598338,
                    entry_shot_type=ShotType.GROUP_WIDE,
                    exit_shot_type=ShotType.CLOSE_UP,
                    exit_boundary_side_stability=outgoing_flash,
                    shot_id="9:wide-then-one-frame-closeup",
                )
            ],
            [
                candidate(1, "9", 0.0, shot_id="9:continue"),
                candidate(
                    1,
                    "3",
                    1.0,
                    entry_shot_scale=0.584663,
                    entry_shot_type=ShotType.CLOSE_UP,
                    entry_boundary_side_stability=stable_incoming,
                    shot_id="3:closeup",
                ),
            ],
        ]

        result = select_timeline(
            segments,
            config=continuity_gate_config(
                maximum_boundary_side_scale_delta=0.20,
                maximum_boundary_side_structure_mismatch=0.34,
                maximum_boundary_side_pose_mismatch=0.42,
                maximum_boundary_side_shot_type_mismatch=0.34,
                require_cross_source_boundary_side_stability=True,
            ),
        )

        self.assertEqual([item.source_id for item in result.candidates], ["9", "9"])

    def test_stable_boundary_sides_do_not_ban_intentional_wide_to_closeup(
        self,
    ) -> None:
        stable_wide = BoundarySideStability(
            sample_span_seconds=3.0 / 29.97,
            scale_delta=0.01,
            structure_mismatch=0.02,
            shot_type_mismatch=0.0,
        )
        stable_closeup = BoundarySideStability(
            sample_span_seconds=3.0 / 29.97,
            scale_delta=0.02,
            structure_mismatch=0.03,
            shot_type_mismatch=0.0,
        )
        segments = [
            [
                candidate(
                    0,
                    "wide",
                    0.5,
                    exit_shot_scale=0.16,
                    exit_shot_type=ShotType.GROUP_WIDE,
                    exit_boundary_side_stability=stable_wide,
                )
            ],
            [
                candidate(1, "wide", 0.0),
                candidate(
                    1,
                    "close",
                    1.0,
                    entry_shot_scale=0.68,
                    entry_shot_type=ShotType.CLOSE_UP,
                    entry_boundary_side_stability=stable_closeup,
                ),
            ],
        ]

        result = select_timeline(
            segments,
            config=continuity_gate_config(
                # This test isolates boundary-side stability. Intentional
                # cross-boundary framing policy remains independently tunable.
                maximum_cross_source_scale_delta=None,
                maximum_boundary_side_scale_delta=0.20,
                maximum_boundary_side_structure_mismatch=0.34,
                maximum_boundary_side_pose_mismatch=0.42,
                maximum_boundary_side_shot_type_mismatch=0.34,
                require_cross_source_boundary_side_stability=True,
            ),
        )

        self.assertEqual(
            [item.source_id for item in result.candidates],
            ["wide", "close"],
        )

    def test_closeup_identity_gate_keeps_same_source_when_face_differs(self) -> None:
        outgoing_identity = (1.0, 0.0)
        different_identity = (0.0, 1.0)
        segments = [
            [
                candidate(
                    0,
                    "A",
                    0.5,
                    shot_type=ShotType.CLOSE_UP,
                    exit_face_identity_embedding=outgoing_identity,
                    shot_id="A:shot-1",
                )
            ],
            [
                candidate(
                    1,
                    "A",
                    0.0,
                    shot_type=ShotType.CLOSE_UP,
                    entry_face_identity_embedding=different_identity,
                    shot_id="A:shot-2",
                ),
                candidate(
                    1,
                    "B",
                    1.0,
                    shot_type=ShotType.CLOSE_UP,
                    entry_face_identity_embedding=different_identity,
                    shot_id="B:shot-1",
                ),
            ],
        ]

        result = select_timeline(
            segments,
            config=continuity_gate_config(
                minimum_cross_source_face_similarity=0.48,
            ),
        )

        self.assertEqual([item.source_id for item in result.candidates], ["A", "A"])

    def test_missing_visual_evidence_gate_keeps_same_source(self) -> None:
        segments = [
            [candidate(0, "A", 0.5, shot_id="A:shot-1")],
            [
                candidate(1, "A", 0.0, shot_id="A:shot-2"),
                candidate(1, "B", 1.0, shot_id="B:shot-1"),
            ],
        ]

        result = select_timeline(
            segments,
            config=continuity_gate_config(
                require_cross_source_visual_evidence=True,
            ),
        )

        self.assertEqual([item.source_id for item in result.candidates], ["A", "A"])

    def test_boundary_detector_miss_does_not_fall_back_to_segment_middle(self) -> None:
        middle_layout = (0.1, 0.2, 0.3)
        segments = [
            [
                candidate(
                    0,
                    "A",
                    0.5,
                    exit_shot_scale=0.4,
                    person_layout_embedding=middle_layout,
                    exit_person_layout_embedding=(),
                    shot_id="A:shot-1",
                )
            ],
            [
                candidate(
                    1,
                    "A",
                    0.0,
                    entry_shot_scale=0.4,
                    person_layout_embedding=middle_layout,
                    entry_person_layout_embedding=(),
                    shot_id="A:shot-2",
                ),
                candidate(
                    1,
                    "B",
                    1.0,
                    entry_shot_scale=0.4,
                    person_layout_embedding=middle_layout,
                    entry_person_layout_embedding=(),
                    shot_id="B:shot-1",
                ),
            ],
        ]

        result = select_timeline(
            segments,
            config=continuity_gate_config(
                require_cross_source_visual_evidence=True,
            ),
        )

        self.assertEqual([item.source_id for item in result.candidates], ["A", "A"])

    def test_motion_transition_uses_outgoing_exit_and_incoming_entry(self) -> None:
        segments = [
            [
                candidate(
                    0,
                    "outgoing",
                    0.5,
                    motion=(1.0, 0.0),
                    exit_motion=(-1.0, 0.0),
                )
            ],
            [
                candidate(
                    1,
                    "matches-exit",
                    0.5,
                    motion=(-1.0, 0.0),
                    entry_motion=(-1.0, 0.0),
                ),
                candidate(
                    1,
                    "matches-legacy-entry",
                    0.5,
                    motion=(1.0, 0.0),
                    entry_motion=(1.0, 0.0),
                ),
            ],
        ]

        result = select_timeline(
            segments,
            config=continuity_gate_config(motion_transition_weight=1.0),
        )

        self.assertEqual(result.candidates[1].source_id, "matches-exit")
        self.assertAlmostEqual(result.choices[1].transition_score, 0.0)

    def test_person_layout_gate_keeps_same_source_on_formation_jump(self) -> None:
        compact_layout = (0.0, 0.0, 0.0, 0.0)
        displaced_layout = (1.0, 1.0, 1.0, 1.0)
        segments = [
            [
                candidate(
                    0,
                    "A",
                    0.5,
                    exit_person_layout_embedding=compact_layout,
                    shot_id="A:shot-1",
                )
            ],
            [
                candidate(
                    1,
                    "A",
                    0.0,
                    entry_person_layout_embedding=displaced_layout,
                    shot_id="A:shot-2",
                ),
                candidate(
                    1,
                    "B",
                    1.0,
                    entry_person_layout_embedding=displaced_layout,
                    shot_id="B:shot-1",
                ),
            ],
        ]

        result = select_timeline(
            segments,
            config=continuity_gate_config(
                maximum_cross_source_layout_mismatch=0.2,
            ),
        )

        self.assertEqual([item.source_id for item in result.candidates], ["A", "A"])

    def test_subject_bbox_layout_fallback_is_a_selector_hard_gate(self) -> None:
        # Boundary geometry from the BLACKHOLE run that used to enter every
        # candidate path and fail only in the final timeline review.
        outgoing_box = (0.28125, 0.390625, 0.34375, 0.484375)
        outgoing_centroid = (0.43720701336860657, 0.6299977898597717)
        incoming_box = (
            0.2604166666666667,
            0.16666666666666666,
            0.6927083333333333,
            0.8333333333333334,
        )
        incoming_centroid = (0.5871025919914246, 0.5764979124069214)
        outgoing = candidate(
            0,
            "A",
            0.5,
            exit_subject_bbox_xywh=outgoing_box,
            exit_subject_centroid_xy=outgoing_centroid,
            exit_silhouette_embedding=(0.2, 0.2, 0.2),
            shot_id="A:shot-1",
        )
        same_source = candidate(
            1,
            "A",
            0.0,
            entry_subject_bbox_xywh=incoming_box,
            entry_subject_centroid_xy=incoming_centroid,
            entry_silhouette_embedding=(0.2, 0.2, 0.2),
            shot_id="A:shot-2",
        )
        tempting_switch = candidate(
            1,
            "B",
            1.0,
            entry_subject_bbox_xywh=incoming_box,
            entry_subject_centroid_xy=incoming_centroid,
            entry_silhouette_embedding=(0.2, 0.2, 0.2),
            shot_id="B:shot-1",
        )

        mismatch = cross_source_layout_mismatch(outgoing, tempting_switch)
        self.assertAlmostEqual(mismatch or 0.0, 0.6039453493532286)
        result = select_timeline(
            ((outgoing,), (same_source, tempting_switch)),
            config=continuity_gate_config(
                maximum_cross_source_layout_mismatch=0.32,
                require_cross_source_visual_evidence=True,
            ),
        )

        self.assertEqual([item.source_id for item in result.candidates], ["A", "A"])

    def test_incomplete_subject_bbox_remains_missing_layout_evidence(self) -> None:
        outgoing = candidate(
            0,
            "A",
            0.5,
            exit_subject_bbox_xywh=(0.2, 0.2, 0.4, 0.6),
            shot_id="A:shot-1",
        )
        incoming = candidate(1, "B", 1.0, shot_id="B:shot-1")

        self.assertIsNone(cross_source_layout_mismatch(outgoing, incoming))

    def test_formation_assignment_detects_left_right_pose_exchange(self) -> None:
        original = (
            formation_person(0.27, raised_arms=False),
            formation_person(0.73, raised_arms=True),
        )
        unchanged = (
            formation_person(0.27, raised_arms=False),
            formation_person(0.73, raised_arms=True),
        )
        exchanged = (
            formation_person(0.27, raised_arms=True),
            formation_person(0.73, raised_arms=False),
        )

        unchanged_mismatch = formation_assignment_mismatch(original, unchanged)
        exchanged_mismatch = formation_assignment_mismatch(original, exchanged)

        self.assertAlmostEqual(unchanged_mismatch or 0.0, 0.0)
        self.assertIsNotNone(exchanged_mismatch)
        assert exchanged_mismatch is not None
        self.assertGreater(exchanged_mismatch, 0.30)

    def test_formation_assignment_penalizes_unmatched_people(self) -> None:
        six = tuple(formation_person(0.1 + index * 0.16) for index in range(6))
        four = six[1:5]

        mismatch = formation_assignment_mismatch(six, four)

        self.assertIsNotNone(mismatch)
        assert mismatch is not None
        self.assertGreaterEqual(mismatch, 2 * 0.82 / 6)

    def test_formation_gate_blocks_cross_source_pose_exchange(self) -> None:
        original = (
            formation_person(0.27, raised_arms=False),
            formation_person(0.73, raised_arms=True),
        )
        exchanged = (
            formation_person(0.27, raised_arms=True),
            formation_person(0.73, raised_arms=False),
        )
        segments = [
            [
                candidate(
                    0,
                    "A",
                    0.5,
                    exit_formation_people=original,
                    shot_id="A:shot-1",
                )
            ],
            [
                candidate(
                    1,
                    "A",
                    0.0,
                    entry_formation_people=original,
                    shot_id="A:shot-2",
                ),
                candidate(
                    1,
                    "B",
                    1.0,
                    entry_formation_people=exchanged,
                    shot_id="B:shot-1",
                ),
            ],
        ]

        result = select_timeline(
            segments,
            config=continuity_gate_config(
                maximum_cross_source_formation_mismatch=0.30,
            ),
        )

        self.assertEqual([item.source_id for item in result.candidates], ["A", "A"])

    def test_pose_model_miss_uses_structure_fallback(self) -> None:
        segments = [
            [
                candidate(
                    0,
                    "outgoing",
                    0.5,
                    exit_pose_embedding=(0.2, 0.2),
                    exit_structure_embedding=(0.0, 0.0),
                )
            ],
            [
                candidate(
                    1,
                    "smooth-fallback",
                    0.5,
                    entry_structure_embedding=(0.0, 0.0),
                ),
                candidate(
                    1,
                    "jump-fallback",
                    0.5,
                    entry_structure_embedding=(1.0, 1.0),
                ),
            ],
        ]
        result = select_timeline(
            segments,
            config=TimelineConfig(
                quality_weight=0.0,
                sync_weight=0.0,
                composition_weight=0.0,
                boundary_weight=0.0,
                cut_penalty=0.0,
                early_cut_penalty=0.0,
                pose_transition_weight=1.0,
                silhouette_transition_weight=0.0,
                motion_transition_weight=0.0,
                scale_transition_weight=0.0,
                color_transition_weight=0.0,
                face_geometry_transition_weight=0.0,
                minimum_hold_segments=1,
            ),
        )
        self.assertEqual(result.candidates[1].source_id, "smooth-fallback")

    def test_group_silhouette_continuity_prefers_matching_formation(self) -> None:
        segments = [
            [candidate(0, "outgoing", 0.5, exit_silhouette_embedding=(0.0, 0.0))],
            [
                candidate(
                    1,
                    "same-formation",
                    0.5,
                    entry_silhouette_embedding=(0.0, 0.0),
                ),
                candidate(
                    1,
                    "different-formation",
                    0.5,
                    entry_silhouette_embedding=(1.0, 1.0),
                ),
            ],
        ]
        result = select_timeline(
            segments,
            config=TimelineConfig(
                quality_weight=0.0,
                sync_weight=0.0,
                composition_weight=0.0,
                boundary_weight=0.0,
                cut_penalty=0.0,
                early_cut_penalty=0.0,
                pose_transition_weight=0.0,
                silhouette_transition_weight=1.0,
                motion_transition_weight=0.0,
                scale_transition_weight=0.0,
                color_transition_weight=0.0,
                face_geometry_transition_weight=0.0,
                minimum_hold_segments=1,
            ),
        )
        self.assertEqual(result.candidates[1].source_id, "same-formation")

        # Missing segmentation is neutral, not an eligibility gate or penalty.
        missing = select_timeline(
            [
                segments[0],
                [
                    candidate(1, "missing-but-better", 0.9),
                    candidate(
                        1,
                        "same-formation",
                        0.1,
                        entry_silhouette_embedding=(0.0, 0.0),
                    ),
                ],
            ],
            config=TimelineConfig(
                quality_weight=1.0,
                sync_weight=0.0,
                composition_weight=0.0,
                boundary_weight=0.0,
                cut_penalty=0.0,
                early_cut_penalty=0.0,
                pose_transition_weight=0.0,
                silhouette_transition_weight=5.0,
                motion_transition_weight=0.0,
                scale_transition_weight=0.0,
                color_transition_weight=0.0,
                face_geometry_transition_weight=0.0,
                minimum_hold_segments=1,
            ),
        )
        self.assertEqual(missing.candidates[1].source_id, "missing-but-better")

    def test_cut_penalizes_outgoing_to_incoming_color_discontinuity(self) -> None:
        segments = [
            [
                candidate(
                    0,
                    "A",
                    0.5,
                    exit_color_embedding=(0.0, 0.0, 0.0),
                )
            ],
            [
                candidate(
                    1,
                    "B",
                    0.5,
                    entry_color_embedding=(1.0, 1.0, 1.0),
                )
            ],
        ]
        config = TimelineConfig(
            quality_weight=0.0,
            sync_weight=0.0,
            composition_weight=0.0,
            boundary_weight=0.0,
            cut_penalty=0.0,
            early_cut_penalty=0.0,
            pose_transition_weight=0.0,
            motion_transition_weight=0.0,
            scale_transition_weight=0.0,
            color_transition_weight=0.75,
            minimum_hold_segments=1,
        )

        result = select_timeline(segments, config=config)

        self.assertAlmostEqual(result.choices[1].transition_score, -0.75)

    def test_face_geometry_uses_worst_consecutive_frame_pair(self) -> None:
        stable = face_geometry()
        shifted = face_geometry(center_x=0.72, center_y=0.54, eye_scale=0.18)

        mismatch = face_geometry_minimax_mismatch(
            (stable, stable, stable),
            (stable, stable, shifted),
        )

        self.assertGreater(mismatch, 0.35)
        self.assertEqual(
            face_geometry_minimax_mismatch((), (stable, stable, stable)),
            0.0,
        )

    def test_global_dp_prefers_smoother_face_window_without_forcing_it(self) -> None:
        stable = face_geometry()
        shifted = face_geometry(center_x=0.75, center_y=0.55, eye_scale=0.19)
        segments = [
            [
                candidate(
                    0,
                    "outgoing",
                    0.5,
                    exit_face_geometry_window=(stable, stable, stable),
                )
            ],
            [
                candidate(
                    1,
                    "smooth",
                    0.5,
                    entry_face_geometry_window=(stable, stable, stable),
                ),
                candidate(
                    1,
                    "jump",
                    0.5,
                    entry_face_geometry_window=(shifted, shifted, shifted),
                ),
            ],
        ]
        config = TimelineConfig(
            quality_weight=0.0,
            sync_weight=0.0,
            composition_weight=0.0,
            boundary_weight=0.0,
            cut_penalty=0.0,
            early_cut_penalty=0.0,
            pose_transition_weight=0.0,
            motion_transition_weight=0.0,
            scale_transition_weight=0.0,
            color_transition_weight=0.0,
            face_geometry_transition_weight=1.0,
            minimum_hold_segments=1,
        )

        result = select_timeline(segments, config=config)

        self.assertEqual(result.candidates[1].source_id, "smooth")

        # Missing face geometry is neutral rather than ineligible or penalized.
        missing_wins = select_timeline(
            [
                segments[0],
                [
                    candidate(1, "missing", 1.0),
                    candidate(
                        1,
                        "smooth",
                        0.1,
                        entry_face_geometry_window=(stable, stable, stable),
                    ),
                ],
            ],
            config=TimelineConfig(
                quality_weight=1.0,
                sync_weight=0.0,
                composition_weight=0.0,
                boundary_weight=0.0,
                cut_penalty=0.0,
                early_cut_penalty=0.0,
                pose_transition_weight=0.0,
                motion_transition_weight=0.0,
                scale_transition_weight=0.0,
                color_transition_weight=0.0,
                face_geometry_transition_weight=5.0,
                minimum_hold_segments=1,
            ),
        )
        self.assertEqual(missing_wins.candidates[1].source_id, "missing")

    def test_member_mode_uses_group_wide_instead_of_wrong_closeup(self) -> None:
        segments = [
            [
                candidate(
                    0,
                    "wrong-closeup",
                    1.0,
                    shot_type=ShotType.CLOSE_UP,
                    target_member=IVEMember.JANGWONYOUNG,
                    target_confidence=0.99,
                    target_crop_available=True,
                ),
                candidate(
                    0,
                    "low-confidence-gaeul",
                    0.95,
                    shot_type=ShotType.MEDIUM,
                    target_member=IVEMember.GAEUL,
                    target_confidence=0.40,
                    target_crop_available=True,
                ),
                candidate(
                    0,
                    "group-wide",
                    0.55,
                    shot_type=ShotType.GROUP_WIDE,
                    is_group_wide=True,
                    shot_scale=0.1,
                ),
            ],
            [
                candidate(
                    1,
                    "verified-gaeul",
                    0.85,
                    shot_type=ShotType.MEDIUM,
                    target_member=IVEMember.GAEUL,
                    target_confidence=0.96,
                    target_crop_available=True,
                    crop_stability_score=0.9,
                ),
                candidate(
                    1,
                    "group-wide",
                    0.50,
                    shot_type=ShotType.GROUP_WIDE,
                    is_group_wide=True,
                    shot_scale=0.1,
                ),
            ],
        ]
        config = TimelineConfig(
            minimum_hold_segments=1,
            cut_penalty=0.0,
            scale_transition_weight=0.0,
        )

        result = select_timeline(
            segments,
            mode=TimelineMode.MEMBER,
            target_member="가을",
            config=config,
        )

        self.assertEqual(result.candidates[0].source_id, "group-wide")
        self.assertEqual(result.candidates[1].source_id, "verified-gaeul")
        self.assertEqual(result.fallback_segment_indices, (0,))

    def test_member_mode_refuses_unsafe_segment(self) -> None:
        segments = [
            [
                candidate(
                    0,
                    "rei-closeup",
                    1.0,
                    shot_type=ShotType.CLOSE_UP,
                    target_member=IVEMember.REI,
                    target_confidence=0.99,
                    target_crop_available=True,
                )
            ]
        ]

        with self.assertRaisesRegex(ValueError, "no safe candidate"):
            select_timeline(
                segments,
                mode=TimelineMode.MEMBER,
                target_member=IVEMember.LIZ,
            )


if __name__ == "__main__":
    unittest.main()
