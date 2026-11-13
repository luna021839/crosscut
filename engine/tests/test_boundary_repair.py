from __future__ import annotations

import unittest

from crosscut.boundary_repair import (
    BOUNDARY_REPAIR_POLICY,
    BoundaryCandidateContext,
    BoundaryRepairConfig,
    BoundaryRepairRequest,
    repair_boundary,
)
from crosscut.stage_edit_temporal_pose_motion import (
    TemporalPoseMotionThresholds,
)


def _request(**overrides: object) -> BoundaryRepairRequest:
    values: dict[str, object] = {
        "nominal_boundary_seconds": 1.0,
        "outgoing_segment_start_seconds": 0.0,
        "incoming_segment_end_seconds": 2.0,
        "output_fps": 10.0,
    }
    values.update(overrides)
    return BoundaryRepairRequest(**values)  # type: ignore[arg-type]


class BoundaryRepairTests(unittest.TestCase):
    def test_temporal_pose_motion_policy_is_fully_injected_and_validated(
        self,
    ) -> None:
        thresholds = TemporalPoseMotionThresholds(
            minimum_boundary_motion_magnitude=0.5,
            minimum_endpoint_pose_cosine=0.8,
            minimum_velocity_direction_cosine=0.7,
            minimum_phase_compatibility=0.75,
            pose_norm_epsilon=1e-8,
        )
        settings = BoundaryRepairConfig(
            temporal_pose_motion_thresholds=thresholds,
            temporal_pose_motion_preference_weight=0.4,
        )
        self.assertIs(settings.temporal_pose_motion_thresholds, thresholds)
        self.assertEqual(
            settings.temporal_pose_motion_preference_weight,
            0.4,
        )
        self.assertIsNone(
            BoundaryRepairConfig().temporal_pose_motion_thresholds
        )
        self.assertEqual(
            BoundaryRepairConfig().temporal_pose_motion_preference_weight,
            0.0,
        )
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            BoundaryRepairConfig(
                temporal_pose_motion_preference_weight=-0.01,
            )
        with self.assertRaisesRegex(TypeError, "TemporalPoseMotionThresholds"):
            BoundaryRepairConfig(
                temporal_pose_motion_thresholds=object(),  # type: ignore[arg-type]
            )

    def test_proxy_triggered_partial_reedit_can_search_eight_frames(self) -> None:
        costs = {offset: 0.8 for offset in range(-8, 9)}
        costs[8] = 0.0
        result = repair_boundary(
            _request(
                nominal_boundary_seconds=2.0,
                outgoing_segment_start_seconds=0.0,
                incoming_segment_end_seconds=4.0,
                output_fps=30.0,
            ),
            visual_costs_by_frame_offset=costs,
            config=BoundaryRepairConfig(
                search_radius_frames=8,
                interior_scene_cut_guard_frames=9,
                visual_cost_slack=0.0,
            ),
        )

        self.assertEqual(result.selected_frame_offset, 8)
        self.assertEqual(len(result.candidates), 17)

    def test_equal_candidates_keep_nominal_boundary(self) -> None:
        result = repair_boundary(
            _request(),
            visual_costs_by_frame_offset={offset: 0.2 for offset in range(-3, 4)},
        )

        self.assertEqual(result.selected_frame_offset, 0)
        self.assertEqual(result.selected_boundary_seconds, 1.0)
        self.assertFalse(result.changed)
        self.assertEqual(result.policy, BOUNDARY_REPAIR_POLICY)

    def test_outgoing_interior_camera_cut_has_strict_priority(self) -> None:
        # At the nominal edit, the broadcaster cut at 0.9 remains visible one
        # frame before our cut. Moving one frame earlier aligns our boundary
        # exactly with it and hides the broadcaster cut outside the rendered
        # outgoing portion, even though that candidate has a worse visual cost.
        result = repair_boundary(
            _request(outgoing_scene_cut_times_seconds=(0.9,)),
            visual_costs_by_frame_offset={
                -3: 0.7,
                -2: 0.6,
                -1: 0.1,
                0: 0.0,
                1: 0.0,
                2: 0.8,
                3: 0.9,
            },
            config=BoundaryRepairConfig(visual_cost_slack=0.0),
        )

        self.assertEqual(result.selected_frame_offset, -1)
        self.assertEqual(result.selected.interior_scene_cut_count, 0)
        self.assertTrue(result.camera_cut_avoidance_improved)

    def test_incoming_camera_cut_can_be_hidden_by_moving_later(self) -> None:
        result = repair_boundary(
            _request(incoming_scene_cut_times_seconds=(1.1,)),
            visual_costs_by_frame_offset={
                -3: 0.5,
                -2: 0.4,
                -1: 0.3,
                0: 0.0,
                1: 0.1,
                2: 0.2,
                3: 0.3,
            },
            config=BoundaryRepairConfig(visual_cost_slack=0.0),
        )

        self.assertEqual(result.selected_frame_offset, 1)
        self.assertEqual(result.selected.interior_scene_cut_count, 0)

    def test_when_every_candidate_has_a_cut_it_minimizes_cut_count_first(self) -> None:
        result = repair_boundary(
            _request(
                outgoing_scene_cut_times_seconds=(0.65, 0.75, 0.85, 0.95),
                incoming_scene_cut_times_seconds=(1.05, 1.15, 1.25, 1.35),
            ),
            visual_costs_by_frame_offset={offset: 0.1 for offset in range(-3, 4)},
            config=BoundaryRepairConfig(
                interior_scene_cut_guard_frames=4,
                visual_cost_slack=0.0,
            ),
        )

        minimum = min(
            candidate.interior_scene_cut_count for candidate in result.candidates
        )
        self.assertEqual(result.selected.interior_scene_cut_count, minimum)

    def test_callback_receives_output_and_both_source_local_clocks(self) -> None:
        contexts: list[BoundaryCandidateContext] = []

        def visual_cost(context: BoundaryCandidateContext) -> float:
            contexts.append(context)
            return abs(context.frame_offset - 2) / 10.0

        result = repair_boundary(
            _request(
                outgoing_source_offset_seconds=2.5,
                incoming_source_offset_seconds=-0.25,
            ),
            visual_cost_callback=visual_cost,
            config=BoundaryRepairConfig(visual_cost_slack=0.0),
        )

        self.assertEqual(result.selected_frame_offset, 2)
        nominal = next(item for item in contexts if item.frame_offset == 0)
        self.assertEqual(nominal.boundary_seconds, 1.0)
        self.assertEqual(nominal.outgoing_source_local_seconds, 3.5)
        self.assertEqual(nominal.incoming_source_local_seconds, 0.75)

    def test_soft_beat_penalty_can_choose_near_visual_nominal(self) -> None:
        result = repair_boundary(
            _request(beat_period_seconds=0.5, beat_intensity=1.0),
            visual_costs_by_frame_offset={
                -3: 0.5,
                -2: 0.5,
                -1: 0.099,
                0: 0.10,
                1: 0.5,
                2: 0.5,
                3: 0.5,
            },
            config=BoundaryRepairConfig(
                visual_cost_slack=0.01,
                beat_penalty_weight=0.04,
            ),
        )

        self.assertEqual(result.selected_frame_offset, 0)
        self.assertIn(-1, result.visual_shortlist_frame_offsets)
        self.assertIn(0, result.visual_shortlist_frame_offsets)

    def test_beat_never_overrides_materially_better_visual_candidate(self) -> None:
        result = repair_boundary(
            _request(beat_period_seconds=0.5, beat_intensity=1.0),
            visual_costs_by_frame_offset={
                -3: 0.5,
                -2: 0.5,
                -1: 0.01,
                0: 0.10,
                1: 0.5,
                2: 0.5,
                3: 0.5,
            },
            config=BoundaryRepairConfig(
                visual_cost_slack=0.015,
                beat_penalty_weight=1.0,
            ),
        )

        self.assertEqual(result.selected_frame_offset, -1)
        self.assertEqual(result.visual_shortlist_frame_offsets, (-1,))

    def test_minimum_two_frame_neighbors_filter_edge_offsets(self) -> None:
        result = repair_boundary(
            _request(
                nominal_boundary_seconds=0.3,
                outgoing_segment_start_seconds=0.0,
                incoming_segment_end_seconds=0.8,
            ),
            visual_costs_by_frame_offset={offset: 0.0 for offset in range(-3, 4)},
        )

        self.assertEqual(
            [item.frame_offset for item in result.rejected],
            [-3, -2],
        )
        self.assertGreaterEqual(
            result.selected_boundary_seconds * 10.0,
            2.0,
        )
        self.assertGreaterEqual(
            (0.8 - result.selected_boundary_seconds) * 10.0,
            2.0,
        )

    def test_scene_cut_exactly_at_boundary_is_hidden_not_interior(self) -> None:
        result = repair_boundary(
            _request(
                outgoing_scene_cut_times_seconds=(1.0,),
                incoming_scene_cut_times_seconds=(1.0,),
            ),
            visual_costs_by_frame_offset={offset: abs(offset) for offset in range(-3, 4)},
            config=BoundaryRepairConfig(visual_cost_slack=0.0),
        )

        self.assertEqual(result.selected_frame_offset, 0)
        self.assertEqual(result.selected.interior_scene_cut_count, 0)

    def test_search_cannot_expose_quantized_hidden_cut_beyond_old_guard(self) -> None:
        # Global selection treats this cut one frame after nominal as the same
        # quantized edit. Moving three frames earlier would expose it four frames
        # into the shot, so the local guard must include both allowances.
        result = repair_boundary(
            _request(incoming_scene_cut_times_seconds=(1.1,)),
            visual_costs_by_frame_offset={
                -3: 0.0,
                -2: 0.8,
                -1: 0.8,
                0: 0.1,
                1: 0.8,
                2: 0.8,
                3: 0.8,
            },
            config=BoundaryRepairConfig(visual_cost_slack=0.0),
        )

        exposed = next(
            candidate for candidate in result.candidates if candidate.frame_offset == -3
        )
        self.assertEqual(exposed.incoming_interior_scene_cuts, (1.1,))
        self.assertEqual(result.selected_frame_offset, 1)
        self.assertEqual(result.selected.interior_scene_cut_count, 0)

    def test_missing_precomputed_cost_rejects_only_that_candidate(self) -> None:
        result = repair_boundary(
            _request(),
            visual_costs_by_frame_offset={-1: 0.3, 0: 0.2, 1: 0.1},
            config=BoundaryRepairConfig(visual_cost_slack=0.0),
        )

        self.assertEqual(result.selected_frame_offset, 1)
        self.assertEqual(
            {item.frame_offset for item in result.rejected},
            {-3, -2, 2, 3},
        )
        self.assertTrue(
            all(item.reason == "visual_cost_unavailable" for item in result.rejected)
        )

    def test_requires_exactly_one_cost_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            repair_boundary(_request())
        with self.assertRaisesRegex(ValueError, "exactly one"):
            repair_boundary(
                _request(),
                visual_cost_callback=lambda context: 0.0,
                visual_costs_by_frame_offset={0: 0.0},
            )

    def test_invalid_request_and_costs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly between"):
            _request(nominal_boundary_seconds=0.0)
        with self.assertRaisesRegex(ValueError, "beat intensity requires"):
            _request(beat_intensity=0.5)
        with self.assertRaisesRegex(ValueError, "outside"):
            repair_boundary(
                _request(),
                visual_costs_by_frame_offset={4: 0.1},
            )
        with self.assertRaisesRegex(ValueError, "negative"):
            repair_boundary(
                _request(),
                visual_costs_by_frame_offset={0: -0.1},
            )
        with self.assertRaisesRegex(ValueError, r"at least search_radius_frames \+ 1"):
            BoundaryRepairConfig(
                search_radius_frames=3,
                interior_scene_cut_guard_frames=3,
            )

    def test_result_is_manifest_ready(self) -> None:
        result = repair_boundary(
            _request(),
            visual_costs_by_frame_offset={offset: abs(offset) for offset in range(-3, 4)},
        )

        payload = result.to_dict()
        self.assertEqual(payload["selected_frame_offset"], 0)
        self.assertEqual(payload["selected"]["interior_scene_cut_count"], 0)
        self.assertEqual(len(payload["candidates"]), 7)


if __name__ == "__main__":
    unittest.main()
