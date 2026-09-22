from __future__ import annotations

from dataclasses import replace
import json
import unittest

from crosscut.pose import FormationPersonDescriptor
from crosscut.timeline import (
    BoundarySideStability,
    INTENTIONAL_PHRASE_ANCHOR_TRANSITION_BASIS,
    ShotType,
    TimelineCandidate,
    TimelineChoice,
    TimelineMode,
    TimelineResult,
    cross_source_layout_mismatch,
)
from crosscut.timeline_review import (
    JOINT_BOUNDARY_RESULT_EVIDENCE_AUTHORITY,
    JOINT_BOUNDARY_RESULT_EVIDENCE_CONTRACT,
    TimelineReviewConfig,
    review_timeline,
)
from crosscut.visual_score import VisualTimelineChoice, VisualTimelinePlan


def json_choice(
    index: int,
    source: int | str,
    start: float,
    end: float,
    **kwargs: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "segment_index": index,
        "source_index": source,
        "canonical_master_start": start,
        "canonical_master_end": end,
    }
    payload.update(kwargs)
    return payload


def candidate(
    index: int,
    source: str,
    start: float,
    end: float,
    **kwargs: object,
) -> TimelineCandidate:
    return TimelineCandidate(
        segment_index=index,
        source_id=source,
        start_seconds=start,
        end_seconds=end,
        **kwargs,
    )


def formation_person(center_x: float) -> FormationPersonDescriptor:
    return FormationPersonDescriptor(
        bbox_xywh=(center_x - 0.08, 0.15, 0.16, 0.70),
        centroid_xy=(center_x, 0.50),
        detection_confidence=0.92,
    )


def visual_choice(
    index: int,
    source: int,
    *,
    start: float | None = None,
    duration: float = 1.4,
    **kwargs: object,
) -> VisualTimelineChoice:
    canonical_start = index * duration if start is None else start
    return VisualTimelineChoice(
        segment_index=index,
        source_index=source,
        canonical_master_start=canonical_start,
        canonical_master_end=canonical_start + duration,
        source_local_start=canonical_start,
        source_local_end=canonical_start + duration,
        quality_score=0.8,
        composition_score=0.8,
        boundary_score=0.8,
        shot_scale=0.4,
        shot_type=ShotType.MEDIUM,
        shot_id=f"{source}:shot-{index}",
        **kwargs,
    )


class TimelineReviewCadenceTests(unittest.TestCase):
    def test_reference_shaped_ten_source_timeline_passes(self) -> None:
        choices = []
        sources = tuple(
            source
            for source, count in enumerate((3, 3, 3, 3, 2, 2, 2, 2, 2, 2))
            for _ in range(count)
        )
        for index, source in enumerate(sources):
            choices.append(
                json_choice(
                    index,
                    source,
                    index * 1.4,
                    (index + 1) * 1.4,
                )
            )

        review = review_timeline(
            {"analysis_basis": "test", "choices": choices},
            available_source_count=10,
        )

        self.assertTrue(review.passed)
        self.assertEqual(review.segment_count, 24)
        self.assertEqual(review.boundary_count, 23)
        self.assertEqual(review.source_switch_count, 9)
        self.assertAlmostEqual(review.source_switch_rate, 9 / 23)
        self.assertAlmostEqual(review.source_switches_per_minute, 9 * 60 / 33.6)
        self.assertEqual(review.unique_source_count, 10)
        self.assertEqual(review.required_unique_source_count, 8)
        self.assertEqual(len(review.source_runs), 10)
        self.assertAlmostEqual(review.source_run_median_seconds, 2.8)
        self.assertAlmostEqual(review.longest_source_run_seconds, 4.2)
        self.assertEqual(len(review.visual_shot_durations_seconds), 24)
        self.assertAlmostEqual(review.visual_shot_median_seconds, 1.4)
        self.assertAlmostEqual(review.per_source_share["0"], 3 / 24)
        self.assertGreater(review.weighted_score, 97.0)

    def test_switch_rate_uses_all_comparable_segment_boundaries(self) -> None:
        choices = [
            json_choice(0, "A", 0.0, 1.4),
            json_choice(1, "A", 1.4, 2.8),
            json_choice(2, "B", 2.8, 4.2),
            json_choice(3, "B", 4.2, 5.6),
            json_choice(4, "C", 5.6, 7.0),
        ]

        review = review_timeline(choices)

        self.assertEqual(review.source_switch_count, 2)
        self.assertAlmostEqual(review.source_switch_rate, 0.5)
        self.assertAlmostEqual(review.source_switches_per_minute, 120.0 / 7.0)
        self.assertNotIn(
            "source_switch_rate_outside_reference_band",
            {warning.code for warning in review.warnings},
        )

    def test_actual_visual_shot_and_source_run_medians_are_serialized(self) -> None:
        choices = [
            json_choice(
                index,
                source,
                float(index),
                float(index + 1),
                shot_id=f"shot-{index // 2}",
            )
            for index, source in enumerate(("A", "A", "B", "B", "C", "C"))
        ]

        review = review_timeline(choices)
        payload = review.to_dict()

        self.assertEqual(review.visual_shot_durations_seconds, (2.0, 2.0, 2.0))
        self.assertEqual(review.visual_shot_median_seconds, 2.0)
        self.assertEqual(review.source_run_median_seconds, 2.0)
        self.assertEqual(payload["visual_shot_count"], 3)
        self.assertEqual(payload["visual_shot_durations_seconds"], [2.0, 2.0, 2.0])
        self.assertEqual(payload["visual_shot_median_seconds"], 2.0)
        self.assertEqual(payload["source_run_median_seconds"], 2.0)
        self.assertEqual(payload["source_switches_per_minute"], 20.0)
        self.assertEqual(
            payload["reference_profile"]["fan_source_switches_per_minute"],
            14.09,
        )
        self.assertEqual(
            payload["reference_profile"]["kbs_source_switches_per_minute"],
            15.39,
        )
        self.assertEqual(
            payload["reference_profile"]["hard_min_source_switches_per_minute"],
            12.5,
        )
        self.assertEqual(
            payload["reference_profile"]["hard_source_run_median_seconds"],
            5.5,
        )
        self.assertEqual(payload["reference_profile"]["source_run_p90_seconds"], 7.2)
        json.dumps(payload)

    def test_internal_detector_cuts_split_one_choice_into_actual_visual_shots(
        self,
    ) -> None:
        review = review_timeline(
            [
                json_choice(
                    0,
                    "A",
                    0.0,
                    4.0,
                    shot_id="A:midpoint-shot",
                    internal_scene_cut_master_seconds=[1.0, 2.5],
                )
            ]
        )
        payload = review.to_dict()

        self.assertEqual(review.internal_scene_cut_count, 2)
        self.assertEqual(review.comparable_visual_boundary_count, 2)
        self.assertEqual(
            review.visual_shot_durations_seconds,
            (1.0, 1.5, 1.5),
        )
        self.assertEqual(review.visual_shot_median_seconds, 1.5)
        self.assertEqual(payload["internal_scene_cut_count"], 2)
        self.assertEqual(payload["visual_shot_count"], 3)
        self.assertEqual(
            payload["visual_shot_durations_seconds"],
            [1.0, 1.5, 1.5],
        )

    def test_internal_cut_explains_changed_midpoint_shot_without_double_split(
        self,
    ) -> None:
        review = review_timeline(
            [
                json_choice(
                    0,
                    "A",
                    0.0,
                    4.0,
                    shot_id="A:shot-0",
                    internal_scene_cut_master_seconds=[3.5],
                ),
                json_choice(
                    1,
                    "A",
                    4.0,
                    8.0,
                    shot_id="A:shot-1",
                ),
            ]
        )

        self.assertFalse(review.boundaries[0].visual_boundary)
        self.assertEqual(review.internal_scene_cut_count, 1)
        self.assertEqual(review.comparable_visual_boundary_count, 1)
        self.assertEqual(review.visual_shot_durations_seconds, (3.5, 4.5))

    def test_exact_boundary_shot_change_without_internal_cut_still_splits(
        self,
    ) -> None:
        review = review_timeline(
            [
                json_choice(0, "A", 0.0, 4.0, shot_id="A:shot-0"),
                json_choice(1, "A", 4.0, 8.0, shot_id="A:shot-1"),
            ]
        )

        self.assertTrue(review.boundaries[0].visual_boundary)
        self.assertEqual(review.internal_scene_cut_count, 0)
        self.assertEqual(review.comparable_visual_boundary_count, 1)
        self.assertEqual(review.visual_shot_durations_seconds, (4.0, 4.0))

    def test_slow_visual_shot_median_warns_and_reduces_score(self) -> None:
        choices = [
            json_choice(
                index,
                source,
                index * 3.1,
                (index + 1) * 3.1,
                shot_id=f"shot-{index}",
            )
            for index, source in enumerate(("A", "B", "C", "D"))
        ]

        review = review_timeline(choices)

        self.assertAlmostEqual(review.visual_shot_median_seconds, 3.1)
        self.assertIn(
            "visual_shot_median_above_preferred",
            {warning.code for warning in review.warnings},
        )
        self.assertEqual(review.score_deductions["visual_shot_cadence"], 10.0)

    def test_source_switch_density_enforces_hard_floor_and_preferred_band(
        self,
    ) -> None:
        def density_review(run_duration: float):
            return review_timeline(
                [
                    json_choice(
                        index,
                        source,
                        index * run_duration,
                        (index + 1) * run_duration,
                    )
                    for index, source in enumerate(("A", "B", "C", "D"))
                ]
            )

        below_hard = density_review(4.5)
        shoulder = density_review(3.5)
        preferred = density_review(3.0)
        above_preferred = density_review(2.25)

        self.assertAlmostEqual(below_hard.source_switches_per_minute, 10.0)
        self.assertIn(
            "source_switch_density_below_hard_minimum",
            {failure.code for failure in below_hard.hard_failures},
        )
        self.assertAlmostEqual(shoulder.source_switches_per_minute, 90.0 / 7.0)
        self.assertNotIn(
            "source_switch_density_below_hard_minimum",
            {failure.code for failure in shoulder.hard_failures},
        )
        self.assertIn(
            "source_switch_density_outside_preferred_range",
            {warning.code for warning in shoulder.warnings},
        )
        self.assertAlmostEqual(preferred.source_switches_per_minute, 15.0)
        self.assertNotIn(
            "source_switch_density_outside_preferred_range",
            {warning.code for warning in preferred.warnings},
        )
        self.assertAlmostEqual(above_preferred.source_switches_per_minute, 20.0)
        self.assertIn(
            "source_switch_density_outside_preferred_range",
            {warning.code for warning in above_preferred.warnings},
        )

    def test_source_run_median_and_reference_p90_are_independent_hard_gates(
        self,
    ) -> None:
        median_failure = review_timeline(
            [
                json_choice(index, source, index * 6.0, (index + 1) * 6.0)
                for index, source in enumerate(("A", "B", "C"))
            ]
        )
        durations = (1.0, 1.0, 8.0, 1.0, 1.0)
        cursor = 0.0
        outlier_choices = []
        for index, (source, duration) in enumerate(
            zip(("A", "B", "C", "D", "E"), durations, strict=True)
        ):
            outlier_choices.append(
                json_choice(index, source, cursor, cursor + duration)
            )
            cursor += duration
        outlier_failure = review_timeline(outlier_choices)

        median_codes = {failure.code for failure in median_failure.hard_failures}
        self.assertEqual(median_failure.source_run_median_seconds, 6.0)
        self.assertIn("source_run_median_hard_max_exceeded", median_codes)
        self.assertNotIn("source_run_hard_max_exceeded", median_codes)

        outlier_codes = {failure.code for failure in outlier_failure.hard_failures}
        self.assertEqual(outlier_failure.source_run_median_seconds, 1.0)
        self.assertEqual(outlier_failure.longest_source_run_seconds, 8.0)
        self.assertNotIn("source_run_median_hard_max_exceeded", outlier_codes)
        self.assertIn("source_run_hard_max_exceeded", outlier_codes)

    def test_actual_seconds_drive_runs_shares_ping_pong_and_short_flash(self) -> None:
        choices = [
            json_choice(0, "A", 0.0, 1.0),
            json_choice(1, "A", 1.0, 2.5),
            json_choice(2, "B", 2.5, 3.0),
            json_choice(3, "A", 3.0, 5.5),
            json_choice(4, "C", 5.5, 8.5),
        ]

        review = review_timeline(choices)

        self.assertEqual(
            [run.duration_seconds for run in review.source_runs],
            [2.5, 0.5, 2.5, 3.0],
        )
        self.assertEqual(review.source_duration_seconds["A"], 5.0)
        self.assertAlmostEqual(review.per_source_share["B"], 0.5 / 8.5)
        self.assertEqual(review.short_flash_run_indices, (1,))
        self.assertEqual(len(review.ping_pongs), 1)
        self.assertEqual(review.ping_pongs[0].source_sequence, ("A", "B", "A"))
        self.assertIn(
            "short_source_flash",
            {failure.code for failure in review.hard_failures},
        )
        self.assertIn(
            "a_b_a_ping_pong",
            {warning.code for warning in review.warnings},
        )

    def test_long_hold_diversity_and_dominance_are_explicit_failures(self) -> None:
        choices = [
            json_choice(index, "only", float(index), float(index + 1))
            for index in range(10)
        ]

        review = review_timeline(choices, available_source_count=10)
        codes = {failure.code for failure in review.hard_failures}

        self.assertFalse(review.passed)
        self.assertIn("insufficient_source_diversity", codes)
        self.assertIn("source_run_hard_max_exceeded", codes)
        self.assertIn("dominant_source_share_hard", codes)
        self.assertEqual(review.longest_source_run_seconds, 10.0)
        self.assertEqual(review.source_switch_count, 0)
        self.assertLess(review.weighted_score, 60.0)

    def test_eight_source_gate_activates_when_eight_are_available(self) -> None:
        choices = [
            json_choice(index, index // 2, index * 1.4, (index + 1) * 1.4)
            for index in range(16)
        ]

        review = review_timeline(choices, available_source_count=8)

        self.assertEqual(review.required_unique_source_count, 8)
        self.assertNotIn(
            "insufficient_source_diversity",
            {failure.code for failure in review.hard_failures},
        )

        below_gate = review_timeline(choices[:14], available_source_count=7)
        self.assertIsNone(below_gate.required_unique_source_count)


class TimelineReviewBoundaryTests(unittest.TestCase):
    def test_candidate_boundary_reports_every_available_mismatch_and_scene_risk(
        self,
    ) -> None:
        outgoing = candidate(
            0,
            "A",
            0.0,
            2.5,
            exit_shot_scale=0.10,
            exit_scene_cut_near=True,
            exit_focus_bbox_xywh=(0.05, 0.10, 0.20, 0.30),
            exit_shot_type=ShotType.GROUP_WIDE,
            exit_person_layout_embedding=(0.0, 0.0),
            exit_silhouette_embedding=(0.0, 0.0),
            exit_formation_people=(
                formation_person(0.15),
                formation_person(0.85),
            ),
            exit_pose_embedding=(0.0, 0.0),
        )
        incoming = candidate(
            1,
            "B",
            2.5,
            5.0,
            entry_shot_scale=0.80,
            entry_focus_bbox_xywh=(0.70, 0.60, 0.20, 0.30),
            entry_shot_type=ShotType.GROUP_WIDE,
            entry_person_layout_embedding=(1.0, 1.0),
            entry_silhouette_embedding=(1.0, 1.0),
            entry_formation_people=(
                formation_person(0.45),
                formation_person(0.55),
            ),
            entry_pose_embedding=(1.0, 1.0),
        )

        review = review_timeline((outgoing, incoming))
        boundary = review.boundaries[0]

        self.assertTrue(boundary.source_switch)
        self.assertTrue(boundary.scene_cut_risk)
        self.assertEqual(
            set(boundary.available_metrics),
            {"scale", "focus", "layout", "silhouette", "formation", "pose"},
        )
        self.assertEqual(set(boundary.exceeded_metrics), set(boundary.available_metrics))
        hard_codes = {failure.code for failure in review.hard_failures}
        self.assertIn("scene_cut_risk_boundary", hard_codes)
        for metric in boundary.available_metrics:
            self.assertIn(f"{metric}_mismatch_exceeded", hard_codes)

    def test_non_group_boundary_does_not_apply_raw_formation_hard_gate(
        self,
    ) -> None:
        review = review_timeline(
            (
                candidate(
                    0,
                    "A",
                    0.0,
                    2.5,
                    exit_shot_type=ShotType.MEDIUM,
                    exit_formation_people=(formation_person(0.20),),
                ),
                candidate(
                    1,
                    "B",
                    2.5,
                    5.0,
                    entry_shot_type=ShotType.CLOSE_UP,
                    entry_formation_people=(formation_person(0.80),),
                ),
            )
        )

        boundary = review.boundaries[0]
        self.assertIsNone(boundary.formation_mismatch)
        self.assertNotIn("formation", boundary.available_metrics)
        self.assertNotIn(
            "formation_mismatch_exceeded",
            {failure.code for failure in review.hard_failures},
        )

    def test_same_source_boundary_is_not_falsely_reviewed_as_cross_source(self) -> None:
        choices = (
            candidate(
                0,
                "A",
                0.0,
                2.5,
                exit_scene_cut_near=True,
                exit_shot_scale=0.0,
                exit_pose_embedding=(0.0, 0.0),
            ),
            candidate(
                1,
                "A",
                2.5,
                5.0,
                entry_scene_cut_near=True,
                entry_shot_scale=1.0,
                entry_pose_embedding=(1.0, 1.0),
            ),
        )

        review = review_timeline(choices)

        self.assertFalse(review.boundaries[0].source_switch)
        self.assertFalse(review.boundaries[0].scene_cut_risk)
        self.assertEqual(review.boundaries[0].available_metrics, ())
        self.assertEqual(review.scene_cut_risk_boundary_indices, ())
        self.assertEqual(
            {summary.metric: summary.available_count for summary in review.mismatch_summaries},
            {
                "scale": 0,
                "focus": 0,
                "layout": 0,
                "silhouette": 0,
                "formation": 0,
                "pose": 0,
            },
        )

    def test_missing_optional_evidence_remains_unavailable(self) -> None:
        review = review_timeline(
            (
                candidate(0, "A", 0.0, 2.5),
                candidate(1, "B", 2.5, 5.0),
            )
        )

        summary = {item.metric: item for item in review.mismatch_summaries}
        self.assertEqual(summary["scale"].available_count, 1)
        self.assertEqual(summary["scale"].maximum, 0.0)
        for metric in ("focus", "layout", "silhouette", "formation", "pose"):
            self.assertEqual(summary[metric].available_count, 0)
            self.assertIsNone(summary[metric].mean)
            self.assertIsNone(summary[metric].maximum)
        self.assertNotIn(
            "pose_mismatch_exceeded",
            {failure.code for failure in review.hard_failures},
        )

    def test_visual_plan_uses_precomputed_transition_metrics(self) -> None:
        outgoing = visual_choice(
            0,
            0,
            exit_shot_scale=0.3,
            exit_shot_type=ShotType.GROUP_WIDE,
            face_samples_xywh=(
                (0.40, 0.20, 0.20, 0.30),
                (0.40, 0.20, 0.20, 0.30),
                (0.40, 0.20, 0.20, 0.30),
            ),
            silhouette_samples_xywh=(
                (0.20, 0.10, 0.60, 0.80),
                (0.20, 0.10, 0.60, 0.80),
                (0.20, 0.10, 0.60, 0.80),
            ),
            silhouette_centroid_samples_xy=((0.5, 0.5),) * 3,
        )
        incoming = visual_choice(
            1,
            1,
            entry_shot_scale=0.35,
            entry_shot_type=ShotType.GROUP_WIDE,
            face_samples_xywh=(
                (0.42, 0.20, 0.20, 0.30),
                (0.42, 0.20, 0.20, 0.30),
                (0.42, 0.20, 0.20, 0.30),
            ),
            silhouette_samples_xywh=(
                (0.22, 0.10, 0.60, 0.80),
                (0.22, 0.10, 0.60, 0.80),
                (0.22, 0.10, 0.60, 0.80),
            ),
            silhouette_centroid_samples_xy=((0.52, 0.5),) * 3,
            silhouette_transition_mismatch_from_previous=0.22,
            layout_transition_mismatch_from_previous=0.19,
            formation_transition_mismatch_from_previous=0.18,
            pose_transition_mismatch_from_previous=0.16,
        )
        plan = VisualTimelinePlan((outgoing, incoming), 8.5)

        review = review_timeline(plan)
        boundary = review.boundaries[0]

        self.assertAlmostEqual(boundary.scale_mismatch or 0.0, 0.05)
        self.assertIsNotNone(boundary.focus_mismatch)
        self.assertEqual(boundary.layout_mismatch, 0.19)
        self.assertEqual(boundary.silhouette_mismatch, 0.22)
        self.assertEqual(boundary.formation_mismatch, 0.18)
        self.assertEqual(boundary.pose_mismatch, 0.16)

    def test_highlight_phrase_anchor_audits_bounded_scale_and_focus_contrast(
        self,
    ) -> None:
        outgoing_face = (0.20, 0.20, 0.20, 0.30)
        incoming_face = (0.41, 0.20, 0.20, 0.30)
        outgoing = visual_choice(
            0,
            0,
            duration=2.4,
            exit_shot_scale=0.10,
            face_samples_xywh=(None, None, outgoing_face),
        )
        incoming = visual_choice(
            1,
            1,
            duration=2.4,
            entry_shot_scale=0.68,
            face_samples_xywh=(incoming_face, None, None),
            intentional_transition_anchor_basis=(
                INTENTIONAL_PHRASE_ANCHOR_TRANSITION_BASIS
            ),
            transition_evidence_authority_from_previous=(
                JOINT_BOUNDARY_RESULT_EVIDENCE_AUTHORITY
            ),
            transition_evidence_contract_from_previous=(
                JOINT_BOUNDARY_RESULT_EVIDENCE_CONTRACT
            ),
            scale_transition_mismatch_from_previous=0.58,
            focus_transition_mismatch_from_previous=0.26,
            layout_transition_mismatch_from_previous=0.20,
            silhouette_transition_mismatch_from_previous=0.20,
            pose_transition_mismatch_from_previous=0.10,
        )
        tail = visual_choice(2, 2, duration=2.4)
        plan = VisualTimelinePlan((outgoing, incoming, tail), 8.5)

        review = review_timeline(plan)
        serialized_review = review_timeline(
            json.loads(json.dumps(plan.to_dict()))
        )
        boundary = review.boundaries[0]

        self.assertEqual(
            boundary.authorized_mismatch_metrics,
            ("scale", "focus"),
        )
        self.assertEqual(boundary.exceeded_metrics, ())
        self.assertNotIn(
            "scale_mismatch_exceeded",
            {failure.code for failure in review.hard_failures},
        )
        self.assertNotIn(
            "focus_mismatch_exceeded",
            {failure.code for failure in review.hard_failures},
        )
        self.assertEqual(
            review.mismatch_summaries[0].exceeded_count,
            0,
        )
        authority = boundary.to_dict()["intentional_transition_authority"]
        self.assertTrue(authority["applied"])
        self.assertEqual(
            authority["authorized_mismatch_metrics"],
            ["scale", "focus"],
        )
        serialized_boundary = serialized_review.boundaries[0]
        self.assertEqual(serialized_boundary.authorized_mismatch_metrics, ())
        self.assertEqual(
            serialized_boundary.exceeded_metrics,
            ("scale", "focus"),
        )
        self.assertIn(
            "scale_mismatch_exceeded",
            {failure.code for failure in serialized_review.hard_failures},
        )
        self.assertIn(
            "focus_mismatch_exceeded",
            {failure.code for failure in serialized_review.hard_failures},
        )

        unmarked_review = review_timeline(
            VisualTimelinePlan(
                (outgoing, replace(incoming, intentional_transition_anchor_basis=None), tail),
                8.5,
            )
        )
        self.assertIn(
            "scale_mismatch_exceeded",
            {failure.code for failure in unmarked_review.hard_failures},
        )
        self.assertIn(
            "focus_mismatch_exceeded",
            {failure.code for failure in unmarked_review.hard_failures},
        )

        weak_pose_review = review_timeline(
            VisualTimelinePlan(
                (
                    outgoing,
                    replace(
                        incoming,
                        pose_transition_mismatch_from_previous=0.25,
                    ),
                    tail,
                ),
                8.5,
            )
        )
        self.assertEqual(
            weak_pose_review.boundaries[0].authorized_mismatch_metrics,
            (),
        )
        self.assertIn(
            "scale_mismatch_exceeded",
            {failure.code for failure in weak_pose_review.hard_failures},
        )

    def test_visual_plan_json_round_trip_has_same_boundary_diagnostics(self) -> None:
        plan = VisualTimelinePlan(
            (
                visual_choice(0, 0, exit_shot_scale=0.20),
                visual_choice(
                    1,
                    1,
                    entry_shot_scale=0.75,
                    pose_transition_mismatch_from_previous=0.70,
                    silhouette_transition_mismatch_from_previous=0.60,
                    formation_transition_mismatch_from_previous=0.50,
                ),
            ),
            4.2,
        )

        object_review = review_timeline(plan)
        json_review = review_timeline(plan.to_dict())

        self.assertEqual(
            object_review.boundaries[0].to_dict(),
            json_review.boundaries[0].to_dict(),
        )
        payload = json_review.to_dict()
        json.dumps(payload)
        self.assertEqual(payload["scene_cut_risk_count"], 0)
        self.assertEqual(payload["reference_profile"]["source_switch_ratio"], 0.49)

    def test_candidate_and_serialized_choice_share_subject_box_layout_gate(
        self,
    ) -> None:
        # Values are the boundary-130 geometry from the production GROUP run.
        # Person-layout inference missed, but the persisted silhouettes prove a
        # large screen-space jump and must fail before candidate selection too.
        outgoing_box = (0.28125, 0.390625, 0.34375, 0.484375)
        outgoing_centroid = (0.43720701336860657, 0.6299977898597717)
        incoming_box = (
            0.2604166666666667,
            0.16666666666666666,
            0.6927083333333333,
            0.8333333333333334,
        )
        incoming_centroid = (0.5871025919914246, 0.5764979124069214)

        candidate_pair = (
            candidate(
                0,
                "6",
                0.0,
                2.5,
                exit_subject_bbox_xywh=outgoing_box,
                exit_subject_centroid_xy=outgoing_centroid,
            ),
            candidate(
                1,
                "2",
                2.5,
                5.0,
                entry_subject_bbox_xywh=incoming_box,
                entry_subject_centroid_xy=incoming_centroid,
            ),
        )
        candidate_review = review_timeline(candidate_pair)
        serialized_review = review_timeline(
            VisualTimelinePlan(
                (
                    visual_choice(
                        0,
                        6,
                        start=0.0,
                        duration=2.5,
                        silhouette_samples_xywh=(None, None, outgoing_box),
                        silhouette_centroid_samples_xy=(
                            None,
                            None,
                            outgoing_centroid,
                        ),
                    ),
                    visual_choice(
                        1,
                        2,
                        start=2.5,
                        duration=2.5,
                        silhouette_samples_xywh=(incoming_box, None, None),
                        silhouette_centroid_samples_xy=(
                            incoming_centroid,
                            None,
                            None,
                        ),
                        layout_transition_mismatch_from_previous=None,
                    ),
                ),
                1.0,
            ).to_dict()
        )

        candidate_boundary = candidate_review.boundaries[0]
        serialized_boundary = serialized_review.boundaries[0]
        self.assertEqual(
            cross_source_layout_mismatch(*candidate_pair),
            candidate_boundary.layout_mismatch,
        )
        self.assertAlmostEqual(
            candidate_boundary.layout_mismatch or 0.0,
            0.6039453493532286,
        )
        self.assertEqual(
            candidate_boundary.to_dict(),
            serialized_boundary.to_dict(),
        )
        for review in (candidate_review, serialized_review):
            self.assertIn("layout", review.boundaries[0].exceeded_metrics)
            self.assertIn(
                "layout_mismatch_exceeded",
                {failure.code for failure in review.hard_failures},
            )

    def test_one_frame_flash_has_candidate_and_serialized_hard_gate_parity(
        self,
    ) -> None:
        # The production 32.041s failure: source 9 changes group-wide ->
        # close-up for its final frame, then source 3 starts close-up. Looking
        # only across the source switch sees a safe .092 scale delta.
        outgoing_stability = BoundarySideStability(
            sample_span_seconds=3.0 / 29.97,
            scale_delta=0.6768598338 - 0.1631326356,
            structure_mismatch=0.51,
            pose_mismatch=None,
            shot_type_mismatch=1.0,
        )
        incoming_stability = BoundarySideStability(
            sample_span_seconds=3.0 / 29.97,
            scale_delta=0.02,
            structure_mismatch=0.04,
            pose_mismatch=0.03,
            shot_type_mismatch=0.0,
        )
        candidate_review = review_timeline(
            (
                candidate(
                    32,
                    "9",
                    0.0,
                    2.5,
                    shot_scale=0.18,
                    entry_shot_scale=0.1631326356,
                    exit_shot_scale=0.6768598338,
                    entry_shot_type=ShotType.GROUP_WIDE,
                    exit_shot_type=ShotType.CLOSE_UP,
                    exit_boundary_side_stability=outgoing_stability,
                ),
                candidate(
                    33,
                    "3",
                    2.5,
                    5.0,
                    shot_scale=0.58,
                    entry_shot_scale=0.584663,
                    entry_shot_type=ShotType.CLOSE_UP,
                    entry_boundary_side_stability=incoming_stability,
                ),
            )
        )
        serialized_plan = VisualTimelinePlan(
            (
                visual_choice(
                    32,
                    9,
                    start=0.0,
                    duration=2.5,
                    entry_shot_scale=0.1631326356,
                    exit_shot_scale=0.6768598338,
                    entry_shot_type=ShotType.GROUP_WIDE,
                    exit_shot_type=ShotType.CLOSE_UP,
                    exit_boundary_side_stability=outgoing_stability,
                ),
                visual_choice(
                    33,
                    3,
                    start=2.5,
                    duration=2.5,
                    entry_shot_scale=0.584663,
                    entry_shot_type=ShotType.CLOSE_UP,
                    entry_boundary_side_stability=incoming_stability,
                ),
            ),
            1.0,
        )
        serialized_review = review_timeline(serialized_plan.to_dict())

        candidate_boundary = candidate_review.boundaries[0]
        serialized_boundary = serialized_review.boundaries[0]
        self.assertAlmostEqual(candidate_boundary.scale_mismatch or 0.0, 0.0921968338)
        self.assertEqual(
            candidate_boundary.to_dict(),
            serialized_boundary.to_dict(),
        )
        self.assertEqual(
            set(candidate_boundary.outgoing_side_stability_exceeded_metrics),
            {"scale", "structure", "shot_type"},
        )
        self.assertEqual(
            candidate_boundary.incoming_side_stability_exceeded_metrics,
            (),
        )
        for review in (candidate_review, serialized_review):
            self.assertEqual(review.outgoing_boundary_side_instability_indices, (0,))
            self.assertEqual(review.incoming_boundary_side_instability_indices, ())
            self.assertIn(
                "outgoing_boundary_side_instability",
                {failure.code for failure in review.hard_failures},
            )
            self.assertGreater(
                review.score_deductions["boundary_side_stability"],
                0.0,
            )
        payload = serialized_review.to_dict()
        self.assertEqual(payload["boundary_side_instability_count"], 1)
        self.assertEqual(
            payload["boundaries"][0]["boundary_side_stability"]["outgoing"][
                "exceeded_metrics"
            ],
            ["scale", "structure", "shot_type"],
        )

    def test_incoming_boundary_side_instability_is_gated_symmetrically(self) -> None:
        unsafe_incoming = BoundarySideStability(
            sample_span_seconds=3.0 / 29.97,
            scale_delta=0.48,
            structure_mismatch=0.50,
            pose_mismatch=0.60,
            shot_type_mismatch=2.0 / 3.0,
        )
        review = review_timeline(
            (
                candidate(0, "A", 0.0, 2.5, exit_shot_scale=0.6),
                candidate(
                    1,
                    "B",
                    2.5,
                    5.0,
                    entry_shot_scale=0.6,
                    entry_boundary_side_stability=unsafe_incoming,
                ),
            )
        )

        self.assertEqual(review.outgoing_boundary_side_instability_indices, ())
        self.assertEqual(review.incoming_boundary_side_instability_indices, (0,))
        self.assertEqual(
            set(
                review.boundaries[0].incoming_side_stability_exceeded_metrics
            ),
            {"scale", "structure", "pose", "shot_type"},
        )
        self.assertIn(
            "incoming_boundary_side_instability",
            {failure.code for failure in review.hard_failures},
        )


class TimelineReviewInputAndValidationTests(unittest.TestCase):
    def test_timeline_result_unwraps_timeline_choice_candidates(self) -> None:
        candidates = (
            candidate(0, "A", 0.0, 1.4),
            candidate(1, "A", 1.4, 2.8),
            candidate(2, "B", 2.8, 4.2),
        )
        choices = tuple(
            TimelineChoice(
                candidate=item,
                emission_score=1.0,
                transition_score=0.0,
                cumulative_score=float(index + 1),
                used_group_wide_fallback=False,
            )
            for index, item in enumerate(candidates)
        )
        result = TimelineResult(
            mode=TimelineMode.GENERAL,
            target_member=None,
            choices=choices,
            total_score=3.0,
        )

        review = review_timeline(result)

        self.assertEqual(review.segment_count, 3)
        self.assertEqual(review.source_switch_count, 1)
        self.assertEqual([run.source_id for run in review.source_runs], ["A", "B"])

    def test_structural_gaps_overlaps_and_bad_duration_are_hard_failures(self) -> None:
        choices = [
            json_choice(0, "A", 0.0, 1.0),
            json_choice(1, "B", 1.2, 2.0),
            json_choice(2, "C", 1.8, 1.8),
        ]

        review = review_timeline(choices)
        codes = {failure.code for failure in review.hard_failures}

        self.assertIn("timeline_gap", codes)
        self.assertIn("timeline_overlap", codes)
        self.assertIn("non_positive_segment_duration", codes)
        self.assertEqual(review.score_deductions["structural_integrity"], 25.0)

    def test_empty_timeline_is_a_serializable_zero_score_failure(self) -> None:
        review = review_timeline({"choices": []})

        self.assertFalse(review.passed)
        self.assertEqual(review.weighted_score, 0.0)
        self.assertEqual(review.hard_failures[0].code, "empty_timeline")
        json.dumps(review.to_dict())

    def test_invalid_input_and_config_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            review_timeline({"source_index": 0})
        with self.assertRaises(ValueError):
            review_timeline([], available_source_count=0)
        with self.assertRaises(ValueError):
            TimelineReviewConfig(
                preferred_min_source_run_seconds=6.0,
                preferred_max_source_run_seconds=5.5,
            )
        with self.assertRaises(ValueError):
            TimelineReviewConfig(mismatch_weights={"scale": 1.0})


if __name__ == "__main__":
    unittest.main()
