from __future__ import annotations

import unittest

import numpy as np

from crosscut.ive import EditMode
from crosscut.timeline import (
    ShotType,
    TimelineCandidate,
    TimelineConfig,
    TimelineMode,
    face_geometry_minimax_mismatch,
    select_timeline,
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

    def test_product_edit_mode_is_accepted(self) -> None:
        result = select_timeline(
            [[candidate(0, "A", 0.8)]], mode=EditMode.GROUP
        )

        self.assertEqual(result.mode, TimelineMode.GENERAL)

    def test_internal_camera_cut_does_not_receive_continuity_bonus(self) -> None:
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

        self.assertEqual(result.choices[1].transition_score, -0.2)

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
