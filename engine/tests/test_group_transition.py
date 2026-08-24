from __future__ import annotations

from dataclasses import replace
import unittest

from crosscut.face_match import FaceMatchPlan, FaceMatchSegment
from crosscut.group_transition import GroupTransitionPlan, build_group_transition_plan
from crosscut.render import (
    NormalizedCrop,
    PanCropKeyframe,
    PanCropTrack,
    RenderTransition,
    TransitionKind,
)
from crosscut.timeline import ShotType
from crosscut.stage_edit_transition_authority import (
    RENDERER_OWNED_TRANSITION_AUTHORITY,
    StageEditTransitionAuthority,
)
from crosscut.visual_score import VisualTimelineChoice, VisualTimelinePlan


_GEOMETRY = (0.45, 0.35, 0.55, 0.35, 0.10, 0.0, 0.0, 0.55, 1.0, 0.0)
_BOX = (0.38, 0.20, 0.24, 0.32)
_IDENTITY = (1.0,) + (0.0,) * 127
_POSE = (0.1,) * 60


def _choice(index: int, source: int) -> VisualTimelineChoice:
    start = float(index)
    end = start + 1.0
    kwargs = {}
    if index == 0:
        kwargs = {
            "dissolve_exit_face_geometry_window": (_GEOMETRY,) * 3,
            "dissolve_exit_face_box_window": (_BOX,) * 3,
            "dissolve_exit_face_identity_window": (_IDENTITY,) * 3,
            "dissolve_exit_pose_embedding_window": (_POSE,) * 3,
            "dissolve_exit_pose_backend_window": (
                "opencv_zoo_mediapipe_pose_2023mar",
            )
            * 3,
            "dissolve_exit_pose_confidence_window": (0.9,) * 3,
        }
    else:
        kwargs = {
            "dissolve_entry_face_geometry_window": (_GEOMETRY,) * 3,
            "dissolve_entry_face_box_window": (_BOX,) * 3,
            "dissolve_entry_face_identity_window": (_IDENTITY,) * 3,
            "dissolve_entry_pose_embedding_window": (_POSE,) * 3,
            "dissolve_entry_pose_backend_window": (
                "opencv_zoo_mediapipe_pose_2023mar",
            )
            * 3,
            "dissolve_entry_pose_confidence_window": (0.9,) * 3,
        }
    return VisualTimelineChoice(
        segment_index=index,
        source_index=source,
        canonical_master_start=start,
        canonical_master_end=end,
        source_local_start=start,
        source_local_end=end,
        quality_score=0.8,
        composition_score=0.8,
        boundary_score=0.8,
        shot_scale=0.4,
        shot_type=ShotType.MEDIUM,
        shot_id=f"{source}:shot-0",
        **kwargs,
    )


def _face_match(choices: tuple[VisualTimelineChoice, ...]) -> FaceMatchPlan:
    crop = NormalizedCrop(0.10, 0.10, 0.80, 0.80)
    segments = tuple(
        FaceMatchSegment(
            segment_index=choice.segment_index,
            source_index=choice.source_index,
            applied=True,
            fallback_reason=None,
            observed_face_height=0.32,
            target_face_height=0.32,
            required_zoom=1.25,
            applied_zoom=1.25,
            alignment_subject="face",
            pan_crop=PanCropTrack(
                (
                    PanCropKeyframe(0.0, crop),
                    PanCropKeyframe(0.5, crop),
                    PanCropKeyframe(1.0, crop),
                )
            ),
        )
        for choice in choices
    )
    return FaceMatchPlan(segments, 0.32, 1.35, (0.5, 0.42))


class GroupTransitionTests(unittest.TestCase):
    def test_transition_counts_separate_visible_cuts_from_same_source_holds(self) -> None:
        plan = GroupTransitionPlan(
            (
                RenderTransition(
                    TransitionKind.FACE_DISSOLVE,
                    11,
                    "safe_face_evidence",
                    0.1,
                    "A",
                ),
                RenderTransition(tier="B", reason="aligned_hard_cut"),
                RenderTransition(tier="C", reason="ordinary_hard_cut"),
                RenderTransition(tier="D", reason="same_source_hold"),
            )
        )

        payload = plan.to_dict(30.0)

        self.assertEqual(plan.dissolve_count, 1)
        self.assertEqual(plan.introduced_hard_cut_count, 2)
        self.assertEqual(plan.same_source_hold_count, 1)
        self.assertEqual(plan.hard_cut_boundary_count, 3)
        self.assertEqual(payload["count_basis"], "transition_boundaries")
        self.assertEqual(payload["introduced_hard_cut_count"], 2)
        self.assertEqual(payload["same_source_hold_count"], 1)
        self.assertEqual(payload["hard_cut_boundary_count"], 3)
        self.assertNotIn("hard_cut_count", payload)

    def test_three_safe_future_face_pairs_promote_to_tier_a_dissolve(self) -> None:
        choices = (_choice(0, 0), _choice(1, 1))
        result = build_group_transition_plan(
            visual_timeline=VisualTimelinePlan(choices, 1.0),
            face_match_plan=_face_match(choices),
            source_durations_seconds=(3.0, 3.0),
            fps=30.0,
        )

        transition = result.transitions[0]
        self.assertEqual(transition.kind, TransitionKind.FACE_DISSOLVE)
        self.assertEqual(transition.frame_count, 11)
        self.assertEqual(transition.tier, "A")
        self.assertAlmostEqual(transition.face_mismatch or 0.0, 0.0)
        self.assertAlmostEqual(
            transition.appearance_min_cosine_similarity or 0.0,
            1.0,
        )
        self.assertAlmostEqual(transition.future_pose_max_mismatch or 0.0, 0.0)
        payload = result.to_dict(30.0)
        self.assertEqual(
            StageEditTransitionAuthority(
                mode=RENDERER_OWNED_TRANSITION_AUTHORITY
            ).to_dict(),
            payload["transition_authority"],
        )
        self.assertEqual(
            ["hard_cut", "face_dissolve"],
            payload["transition_authority"]["renderer_allowed_kinds"],
        )
        self.assertFalse(
            payload["transition_authority"][
                "zoom_match_whip_pan_render_effects_authorized"
            ]
        )

    def test_missing_future_evidence_stays_tier_b_aligned_hard_cut(self) -> None:
        choices = (_choice(0, 0), replace(_choice(1, 1), dissolve_entry_face_geometry_window=()))
        result = build_group_transition_plan(
            visual_timeline=VisualTimelinePlan(choices, 1.0),
            face_match_plan=_face_match(choices),
            source_durations_seconds=(3.0, 3.0),
            fps=30.0,
        )

        transition = result.transitions[0]
        self.assertEqual(transition.kind, TransitionKind.HARD_CUT)
        self.assertEqual(transition.tier, "B")
        self.assertEqual(transition.reason, "future_face_evidence_missing_or_ambiguous")

    def test_future_body_mismatch_vetoes_blend(self) -> None:
        first, second = _choice(0, 0), _choice(1, 1)
        second = replace(
            second,
            dissolve_entry_pose_embedding_window=((1.0,) * 60,) * 3,
        )
        result = build_group_transition_plan(
            visual_timeline=VisualTimelinePlan((first, second), 1.0),
            face_match_plan=_face_match((first, second)),
            source_durations_seconds=(3.0, 3.0),
            fps=30.0,
        )

        self.assertEqual(result.transitions[0].kind, TransitionKind.HARD_CUT)
        self.assertEqual(result.transitions[0].reason, "future_body_pose_mismatch")
        self.assertGreater(result.transitions[0].future_pose_max_mismatch or 0.0, 0.8)

    def test_missing_future_body_evidence_stays_tier_b(self) -> None:
        first, second = _choice(0, 0), _choice(1, 1)
        second = replace(
            second,
            dissolve_entry_pose_embedding_window=(),
            dissolve_entry_pose_backend_window=(),
            dissolve_entry_pose_confidence_window=(),
        )
        result = build_group_transition_plan(
            visual_timeline=VisualTimelinePlan((first, second), 1.0),
            face_match_plan=_face_match((first, second)),
            source_durations_seconds=(3.0, 3.0),
            fps=30.0,
        )

        self.assertEqual(result.transitions[0].kind, TransitionKind.HARD_CUT)
        self.assertEqual(result.transitions[0].tier, "B")
        self.assertEqual(result.transitions[0].reason, "future_body_evidence_unavailable")

    def test_different_face_appearance_never_promotes_to_dissolve(self) -> None:
        first, second = _choice(0, 0), _choice(1, 1)
        different_identity = (0.0, 1.0) + (0.0,) * 126
        second = replace(
            second,
            dissolve_entry_face_identity_window=(different_identity,) * 3,
        )
        result = build_group_transition_plan(
            visual_timeline=VisualTimelinePlan((first, second), 1.0),
            face_match_plan=_face_match((first, second)),
            source_durations_seconds=(3.0, 3.0),
            fps=30.0,
        )

        transition = result.transitions[0]
        self.assertEqual(transition.kind, TransitionKind.HARD_CUT)
        self.assertEqual(transition.tier, "B")
        self.assertEqual(transition.reason, "future_face_appearance_mismatch")
        self.assertAlmostEqual(
            transition.appearance_min_cosine_similarity or 0.0,
            0.0,
        )

    def test_silhouette_alignment_is_tier_b_and_never_face_dissolve(self) -> None:
        choices = (_choice(0, 0), _choice(1, 1))
        matches = _face_match(choices)
        silhouette_matches = FaceMatchPlan(
            (
                replace(matches.segments[0], alignment_subject="silhouette"),
                matches.segments[1],
            ),
            matches.target_face_height,
            matches.maximum_zoom,
            matches.target_screen_position,
        )

        result = build_group_transition_plan(
            visual_timeline=VisualTimelinePlan(choices, 1.0),
            face_match_plan=silhouette_matches,
            source_durations_seconds=(3.0, 3.0),
            fps=30.0,
        )

        self.assertEqual(result.transitions[0].kind, TransitionKind.HARD_CUT)
        self.assertEqual(result.transitions[0].tier, "B")
        self.assertEqual(
            result.transitions[0].reason,
            "silhouette_aligned_hard_cut_only",
        )

    def test_source_end_raw_fallback_and_same_source_receive_distinct_tiers(self) -> None:
        choices = (_choice(0, 0), _choice(1, 1))
        unavailable = build_group_transition_plan(
            visual_timeline=VisualTimelinePlan(choices, 1.0),
            face_match_plan=_face_match(choices),
            source_durations_seconds=(1.2, 3.0),
            fps=30.0,
        )
        raw = build_group_transition_plan(
            visual_timeline=VisualTimelinePlan(choices, 1.0),
            face_match_plan=None,
            source_durations_seconds=(3.0, 3.0),
            fps=30.0,
        )
        held_choices = (_choice(0, 0), _choice(1, 0))
        held = build_group_transition_plan(
            visual_timeline=VisualTimelinePlan(held_choices, 1.0),
            face_match_plan=None,
            source_durations_seconds=(3.0, 3.0),
            fps=30.0,
        )
        internal_cut_choices = (
            _choice(0, 0),
            replace(_choice(1, 0), shot_id="0:shot-1"),
        )
        internal_cut = build_group_transition_plan(
            visual_timeline=VisualTimelinePlan(internal_cut_choices, 1.0),
            face_match_plan=None,
            source_durations_seconds=(3.0, 3.0),
            fps=30.0,
        )

        self.assertEqual(unavailable.transitions[0].tier, "B")
        self.assertEqual(
            unavailable.transitions[0].reason,
            "outgoing_source_handle_unavailable",
        )
        self.assertEqual(raw.transitions[0].tier, "C")
        self.assertEqual(held.transitions[0].tier, "D")
        self.assertEqual(internal_cut.transitions[0].tier, "C")
        self.assertEqual(
            internal_cut.transitions[0].reason,
            "same_source_internal_camera_cut",
        )


if __name__ == "__main__":
    unittest.main()
