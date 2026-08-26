from __future__ import annotations

from types import SimpleNamespace
import unittest

from crosscut.direct_member_professional_gate import (
    DirectMemberProfessionalGateConfig,
    evaluate_direct_member_professional_pre_render_gate,
)
from crosscut.direct_member_transform import direct_member_source_frame_geometry
from crosscut.render import RenderTransition, TransitionKind


class _MotionPlan:
    def __init__(self, *, applied_count: int, accepted: tuple[int, ...] = ()) -> None:
        self.applied_count = applied_count
        self.accepted_boundary_indices = accepted


def _selection(
    *,
    count: int = 20,
    repair_index: int | None = None,
    missing_pose_index: int | None = None,
    opposed_motion_index: int | None = None,
    speed_discontinuity_index: int | None = None,
    phase_discontinuity_index: int | None = None,
):
    config = SimpleNamespace(
        maximum_cross_source_pose_mismatch=0.30,
        maximum_cross_source_silhouette_mismatch=0.42,
        minimum_motion_direction_cosine=0.12,
        maximum_motion_endpoint_shape_mismatch=0.30,
        maximum_motion_speed_ratio_mismatch=0.60,
        maximum_pose_phase_continuation_mismatch=0.34,
        transform_alignment_position_trigger=0.08,
        transform_alignment_scale_trigger=0.06,
        transform_body_alignment_position_trigger=0.12,
        transform_body_alignment_scale_trigger=0.10,
    )
    boundaries = []
    for index in range(count):
        attention = SimpleNamespace(
            basis="unambiguous_dominant_face_bbox_geometry_only"
        )
        repair = (
            SimpleNamespace(
                residual_attention_position_mismatch=0.03,
                residual_attention_scale_mismatch=0.03,
            )
            if index == repair_index
            else None
        )
        motion = SimpleNamespace(
            both_moving=True,
            direction_cosine_audit_only=(
                -0.4 if index == opposed_motion_index else 0.8
            ),
            endpoint_pose_shape_mismatch=0.10,
            motion_speed_ratio_mismatch=(
                0.90 if index == speed_discontinuity_index else 0.10
            ),
            pose_phase_continuation_mismatch=(
                0.80 if index == phase_discontinuity_index else 0.10
            ),
        )
        boundaries.append(
            SimpleNamespace(
                boundary_index=index,
                cut_applied=True,
                outgoing_sample=SimpleNamespace(attention=attention),
                incoming_sample=SimpleNamespace(attention=attention),
                mismatch=SimpleNamespace(
                    dominant_pose_mismatch=(
                        None if index == missing_pose_index else 0.10
                    ),
                    silhouette_mismatch=0.16,
                    attention_position_mismatch=(
                        0.16 if index == repair_index else 0.04
                    ),
                    attention_scale_mismatch=(
                        0.14 if index == repair_index else 0.03
                    ),
                    transform_repair=repair,
                    motion_connection=motion,
                    severe_flags=(),
                    cinematic_bridge=None,
                ),
            )
        )
    return SimpleNamespace(boundaries=tuple(boundaries), config=config)


def _cadence(*, switch_count: int = 20, effects: int = 0, requested=False):
    return {
        "source_switch_count": switch_count,
        "member_pose_dissolve_requested": requested,
        "member_pose_dissolve_count": effects,
        "member_cinematic_bridge_count": 0,
        "pose_matched_dissolves": {
            "eligible_candidate_count": effects,
        },
        "highlight_cut_density": {"passed": True},
    }


class DirectMemberProfessionalGateTests(unittest.TestCase):
    def test_professional_plan_passes_joint_gate(self) -> None:
        result = evaluate_direct_member_professional_pre_render_gate(
            visual_selection=_selection(),
            cadence=_cadence(),
            editorial_motion_plan=_MotionPlan(applied_count=8),
            transitions=tuple(RenderTransition() for _ in range(20)),
            duration_seconds=80.0,
            dense_geometry_review_completed=True,
        )
        self.assertTrue(result.passed, result.to_dict())
        self.assertEqual(result.metrics["visible_switches_per_minute"], 15.0)
        self.assertEqual(result.failure_reasons, ())

    def test_missing_pose_and_opposed_motion_fail_closed(self) -> None:
        result = evaluate_direct_member_professional_pre_render_gate(
            visual_selection=_selection(
                missing_pose_index=3,
                opposed_motion_index=7,
            ),
            cadence=_cadence(),
            editorial_motion_plan=_MotionPlan(applied_count=8),
            transitions=tuple(RenderTransition() for _ in range(20)),
            duration_seconds=80.0,
            dense_geometry_review_completed=True,
        )
        self.assertFalse(result.passed)
        self.assertEqual(
            result.failure_boundary_indices["missing_actual_pose"], (3,)
        )
        self.assertEqual(
            result.failure_boundary_indices[
                "opposed_or_missing_motion_direction"
            ],
            (7,),
        )

    def test_transform_repair_needs_dense_acceptance(self) -> None:
        rejected = evaluate_direct_member_professional_pre_render_gate(
            visual_selection=_selection(repair_index=5),
            cadence=_cadence(),
            editorial_motion_plan=_MotionPlan(applied_count=8),
            transitions=tuple(RenderTransition() for _ in range(20)),
            duration_seconds=80.0,
            dense_geometry_review_completed=True,
        )
        accepted = evaluate_direct_member_professional_pre_render_gate(
            visual_selection=_selection(repair_index=5),
            cadence=_cadence(),
            editorial_motion_plan=_MotionPlan(applied_count=8, accepted=(5,)),
            transitions=tuple(RenderTransition() for _ in range(20)),
            duration_seconds=80.0,
            dense_geometry_review_completed=True,
        )
        self.assertFalse(rejected.passed)
        self.assertIn(
            "required_dense_transform_not_accepted",
            rejected.failure_reasons,
        )
        self.assertTrue(accepted.passed, accepted.to_dict())

    def test_speed_and_pose_phase_discontinuities_fail_closed(self) -> None:
        result = evaluate_direct_member_professional_pre_render_gate(
            visual_selection=_selection(
                speed_discontinuity_index=4,
                phase_discontinuity_index=9,
            ),
            cadence=_cadence(),
            editorial_motion_plan=_MotionPlan(applied_count=8),
            transitions=tuple(RenderTransition() for _ in range(20)),
            duration_seconds=80.0,
            dense_geometry_review_completed=True,
        )

        self.assertFalse(result.passed)
        self.assertEqual(
            result.failure_boundary_indices["incompatible_motion_speed"],
            (4,),
        )
        self.assertEqual(
            result.failure_boundary_indices["incompatible_pose_phase"],
            (9,),
        )

    def test_requested_safe_effect_must_exist_in_materialized_transitions(self) -> None:
        config = DirectMemberProfessionalGateConfig(
            minimum_selected_effect_count_when_requested=1
        )
        rejected = evaluate_direct_member_professional_pre_render_gate(
            visual_selection=_selection(),
            cadence=_cadence(effects=1, requested=True),
            editorial_motion_plan=_MotionPlan(applied_count=8),
            transitions=tuple(RenderTransition() for _ in range(20)),
            duration_seconds=80.0,
            dense_geometry_review_completed=True,
            config=config,
        )
        transitions = [RenderTransition() for _ in range(20)]
        transitions[2] = RenderTransition(
            kind=TransitionKind.FACE_DISSOLVE,
            frame_count=6,
            reason="verified_pose_match",
        )
        accepted = evaluate_direct_member_professional_pre_render_gate(
            visual_selection=_selection(),
            cadence=_cadence(effects=1, requested=True),
            editorial_motion_plan=_MotionPlan(applied_count=8),
            transitions=tuple(transitions),
            duration_seconds=80.0,
            dense_geometry_review_completed=True,
            config=config,
        )
        self.assertIn(
            "materialized_effect_count_disagrees_with_cadence",
            rejected.failure_reasons,
        )
        self.assertTrue(accepted.passed, accepted.to_dict())

    def test_sparse_editorial_camera_fails_even_when_cuts_are_safe(self) -> None:
        result = evaluate_direct_member_professional_pre_render_gate(
            visual_selection=_selection(),
            cadence=_cadence(),
            editorial_motion_plan=_MotionPlan(applied_count=2),
            transitions=tuple(RenderTransition() for _ in range(20)),
            duration_seconds=80.0,
            dense_geometry_review_completed=True,
        )
        self.assertFalse(result.passed)
        self.assertIn(
            "editorial_camera_density_below_professional_floor",
            result.failure_reasons,
        )

    def test_enabled_effects_need_a_visible_song_level_density(self) -> None:
        cadence = _cadence(effects=1, requested=True)
        cadence["pose_matched_dissolves"]["eligible_candidate_count"] = 5
        transitions = [RenderTransition() for _ in range(20)]
        transitions[2] = RenderTransition(
            kind=TransitionKind.FACE_DISSOLVE,
            frame_count=6,
            reason="verified_pose_match",
        )
        result = evaluate_direct_member_professional_pre_render_gate(
            visual_selection=_selection(),
            cadence=cadence,
            editorial_motion_plan=_MotionPlan(applied_count=8),
            transitions=tuple(transitions),
            duration_seconds=80.0,
            dense_geometry_review_completed=True,
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.metrics["required_effect_count"], 2)
        self.assertIn(
            "requested_safe_transition_effect_not_materialized",
            result.failure_reasons,
        )

    def test_native_output_portrait_source_is_used_without_dominating_4k(
        self,
    ) -> None:
        selection = _selection()
        selection.source_frame_geometries = (
            direct_member_source_frame_geometry((2160, 3840)),
            direct_member_source_frame_geometry((1080, 1920)),
        )
        selection.source_durations_seconds = (40.0, 40.0)
        rejected = evaluate_direct_member_professional_pre_render_gate(
            visual_selection=selection,
            cadence=_cadence(),
            editorial_motion_plan=_MotionPlan(applied_count=8),
            transitions=tuple(RenderTransition() for _ in range(20)),
            duration_seconds=80.0,
            dense_geometry_review_completed=True,
        )
        selection.source_durations_seconds = (60.0, 20.0)
        accepted = evaluate_direct_member_professional_pre_render_gate(
            visual_selection=selection,
            cadence=_cadence(),
            editorial_motion_plan=_MotionPlan(applied_count=8),
            transitions=tuple(RenderTransition() for _ in range(20)),
            duration_seconds=80.0,
            dense_geometry_review_completed=True,
        )

        self.assertIn(
            "native_output_portrait_duration_fraction_above_quality_floor",
            rejected.failure_reasons,
        )
        self.assertTrue(accepted.passed, accepted.to_dict())


if __name__ == "__main__":
    unittest.main()
