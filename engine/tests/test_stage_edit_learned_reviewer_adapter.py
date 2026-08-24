from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from crosscut.ive import AspectRatio
from crosscut.render import RenderPlan, RenderSegment, RenderTransition
from crosscut.rendered_proxy_review import (
    ANALYSIS_BASIS as DECODED_PROXY_ANALYSIS_BASIS,
    INTRODUCED_CROSS_SOURCE_DOMAIN,
    NATIVE_SOURCE_CUT_DOMAIN,
)
from crosscut.stage_edit_learned_reviewer import (
    DEFAULT_LEARNED_REVIEWER_SCHEMA,
    LEARNED_REVIEWER_FEATURE_MATERIALIZER_CONTRACT_SHA256,
    LearnedReviewerBoundaryPrediction,
    LearnedReviewerPrediction,
)
from crosscut.stage_edit_learned_reviewer_adapter import (
    LearnedDecodedCandidateReview,
    LearnedReviewerAdapterError,
    augment_stage_edit_repair_review_with_learned_issues,
    learned_prediction_to_stage_edit_repair_issues,
    materialize_decoded_proxy_reviewer_features,
)
from crosscut.stage_edit_repair_loop import (
    StageEditRepairIssue,
    StageEditRepairReview,
)


def _plan() -> RenderPlan:
    return RenderPlan(
        sources=(Path("source-a.mp4"), Path("source-b.mp4")),
        segments=(
            RenderSegment(0, 0.0, 1.0, 0.0),
            RenderSegment(1, 1.0, 1.0, 1.0),
            RenderSegment(1, 2.0, 1.0, 2.0),
        ),
        reference_source=0,
        aspect_ratio=AspectRatio.LANDSCAPE,
        output=Path("candidate.mp4"),
        master_audio_start=4.0,
        width=1920,
        height=1080,
        transitions=(
            RenderTransition(tier="C", reason="introduced"),
            RenderTransition(tier="C", reason="native"),
        ),
        fps=30.0,
    )


def _boundary(
    index: int,
    *,
    cut_frame: int,
    outgoing: int,
    incoming: int,
    domain: str,
) -> dict:
    return {
        "boundary_index": index,
        "cut_frame_index": cut_frame,
        "outgoing_source_index": outgoing,
        "incoming_source_index": incoming,
        "transition_kind": "hard_cut",
        "review_domain": domain,
        "weighted_score": 72.0 - index,
        "discontinuities": {
            "face_scale_discontinuity": 0.10,
            "face_position_discontinuity": 0.20,
            "silhouette_scale_discontinuity": 0.15,
            "silhouette_position_discontinuity": 0.25,
            "structure_shock": 0.30,
            "motion_direction_discontinuity": 0.20,
            "color_shock": 0.10,
            "exposure_discontinuity": 0.05,
        },
        "frame_metrics": [{"frame_index": cut_frame, "sharpness": 0.80}],
        "temporal_integrity": {
            "black_frame_offsets": [],
            "duplicate_or_freeze_detected": False,
        },
        "technical_integrity_issue_reasons": [],
    }


def _review(plan: RenderPlan) -> dict:
    return {
        "schema_version": 3,
        "analysis": DECODED_PROXY_ANALYSIS_BASIS,
        "proxy": {
            "fps": plan.fps,
            "decoded_frame_count": plan.output_frame_count,
            "duration_seconds": plan.render_duration,
            "path": None,
        },
        "passed": True,
        "weighted_score": 71.5,
        "boundary_score": {"issue_fraction": 0.0, "p10": 71.0},
        "boundaries": [
            _boundary(
                0,
                cut_frame=30,
                outgoing=0,
                incoming=1,
                domain=INTRODUCED_CROSS_SOURCE_DOMAIN,
            ),
            _boundary(
                1,
                cut_frame=60,
                outgoing=1,
                incoming=1,
                domain=NATIVE_SOURCE_CUT_DOMAIN,
            ),
        ],
    }


class StageEditLearnedReviewerAdapterTests(unittest.TestCase):
    def test_materializer_binds_clock_plan_evidence_and_missing_masks(self) -> None:
        plan = _plan()
        materialized = materialize_decoded_proxy_reviewer_features(
            candidate_index=7,
            song_id="BANG BANG",
            render_plan=plan,
            decoded_review=_review(plan),
            source_switches_per_minute=20.0,
        )

        self.assertEqual(materialized.boundary_indices, (0, 1))
        self.assertEqual(
            materialized.boundary_domains,
            (INTRODUCED_CROSS_SOURCE_DOMAIN, NATIVE_SOURCE_CUT_DOMAIN),
        )
        self.assertEqual(
            tuple(item.at_seconds for item in materialized.example.boundaries),
            (5.0, 6.0),
        )
        self.assertTrue(materialized.technical_zero_tolerance_passed)
        self.assertEqual(
            len(materialized.example.boundaries[0].features),
            DEFAULT_LEARNED_REVIEWER_SCHEMA.boundary_dimension,
        )
        section_indices = tuple(
            DEFAULT_LEARNED_REVIEWER_SCHEMA.timeline_feature_names.index(name)
            for name in (
                "section_energy_alignment",
                "section_cut_density_alignment",
            )
        )
        self.assertEqual(
            tuple(materialized.example.timeline_features[index] for index in section_indices),
            (None, None),
        )
        pose_index = DEFAULT_LEARNED_REVIEWER_SCHEMA.boundary_feature_names.index(
            "pose_mismatch"
        )
        self.assertIsNone(materialized.example.boundaries[0].features[pose_index])
        audit = materialized.to_audit_dict()
        self.assertEqual(
            audit["feature_materializer_contract_sha256"],
            LEARNED_REVIEWER_FEATURE_MATERIALIZER_CONTRACT_SHA256,
        )
        for key in (
            "render_plan_sha256",
            "decoded_evidence_sha256",
            "feature_snapshot_sha256",
        ):
            self.assertEqual(len(audit[key]), 64)
        self.assertFalse(audit["raw_frames_or_embeddings_persisted"])

    def test_non_feature_selection_fields_cannot_leak_into_model_features(self) -> None:
        plan = _plan()
        baseline_review = _review(plan)
        changed_review = deepcopy(baseline_review)
        changed_review.update(
            {
                "passed": False,
                "weighted_score": 1.0,
                "selection_basis": "must_not_enter_features",
                "professional_reference_gate": {
                    "passed": False,
                    "technical_integrity": {"passed": False, "issue_count": 99},
                },
            }
        )
        baseline = materialize_decoded_proxy_reviewer_features(
            candidate_index=0,
            song_id="BANG BANG",
            render_plan=plan,
            decoded_review=baseline_review,
        )
        changed = materialize_decoded_proxy_reviewer_features(
            candidate_index=0,
            song_id="BANG BANG",
            render_plan=plan,
            decoded_review=changed_review,
        )

        self.assertEqual(baseline.example, changed.example)
        self.assertEqual(
            baseline.feature_snapshot_sha256,
            changed.feature_snapshot_sha256,
        )
        self.assertNotEqual(
            baseline.decoded_evidence_sha256,
            changed.decoded_evidence_sha256,
        )

    def test_materializer_fails_closed_on_clock_source_and_technical_tamper(self) -> None:
        plan = _plan()
        cases: list[tuple[str, dict]] = []
        bad_clock = _review(plan)
        bad_clock["proxy"]["decoded_frame_count"] -= 1
        cases.append(("frame count", bad_clock))
        bad_source = _review(plan)
        bad_source["boundaries"][0]["incoming_source_index"] = 0
        cases.append(("sources", bad_source))
        bad_boolean = _review(plan)
        bad_boolean["boundaries"][0]["temporal_integrity"][
            "duplicate_or_freeze_detected"
        ] = "false"
        cases.append(("boolean", bad_boolean))
        for expected, payload in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(LearnedReviewerAdapterError, expected):
                    materialize_decoded_proxy_reviewer_features(
                        candidate_index=0,
                        song_id="BANG BANG",
                        render_plan=plan,
                        decoded_review=payload,
                    )

    def test_prediction_proposes_only_nontechnical_introduced_windows(self) -> None:
        plan = _plan()
        materialized = materialize_decoded_proxy_reviewer_features(
            candidate_index=2,
            song_id="BANG BANG",
            render_plan=plan,
            decoded_review=_review(plan),
        )
        first, second = materialized.example.boundaries
        prediction = LearnedReviewerPrediction(
            example_id=materialized.example.example_id,
            learned_quality_probability=0.25,
            learned_quality_score=25.0,
            model_recommendation_passed=False,
            technical_zero_tolerance_passed=True,
            review_passed=False,
            boundary_predictions=(
                LearnedReviewerBoundaryPrediction(
                    boundary_id=first.boundary_id,
                    at_seconds=first.at_seconds,
                    issue_probability=0.90,
                    cause_probabilities={
                        "visual_discontinuity": 0.80,
                        "technical_integrity": 0.99,
                    },
                    technical_error_codes=(),
                    technical_hard_failure=False,
                ),
                LearnedReviewerBoundaryPrediction(
                    boundary_id=second.boundary_id,
                    at_seconds=second.at_seconds,
                    issue_probability=0.99,
                    cause_probabilities={"motion_mismatch": 0.90},
                    technical_error_codes=(),
                    technical_hard_failure=False,
                ),
            ),
        )
        issues, indices = learned_prediction_to_stage_edit_repair_issues(
            materialized,
            prediction,
        )

        self.assertEqual(indices, (0,))
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].causes, ("visual_discontinuity",))
        self.assertEqual((issues[0].start_seconds, issues[0].end_seconds), (4.8, 5.2))

    def test_repair_review_augmentation_cannot_change_hard_objectives(self) -> None:
        plan = _plan()
        materialized = materialize_decoded_proxy_reviewer_features(
            candidate_index=2,
            song_id="BANG BANG",
            render_plan=plan,
            decoded_review=_review(plan),
        )
        boundary = materialized.example.boundaries[0]
        prediction = LearnedReviewerPrediction(
            example_id=materialized.example.example_id,
            learned_quality_probability=0.20,
            learned_quality_score=20.0,
            model_recommendation_passed=False,
            technical_zero_tolerance_passed=True,
            review_passed=False,
            boundary_predictions=tuple(
                LearnedReviewerBoundaryPrediction(
                    boundary_id=item.boundary_id,
                    at_seconds=item.at_seconds,
                    issue_probability=0.90 if item is boundary else 0.0,
                    cause_probabilities={"visual_discontinuity": 0.9},
                    technical_error_codes=(),
                    technical_hard_failure=False,
                )
                for item in materialized.example.boundaries
            ),
        )
        issues, indices = learned_prediction_to_stage_edit_repair_issues(
            materialized, prediction
        )
        learned = LearnedDecodedCandidateReview(
            2, materialized, prediction, issues, indices
        )
        original_issue = StageEditRepairIssue(
            "deterministic", 4.0, 4.1, ("decoded",)
        )
        deterministic = StageEditRepairReview(
            decoded_objective=0.7,
            professional_objective=0.8,
            cadence_objective=0.6,
            technical_errors=("clock_error",),
            issue_windows=(original_issue,),
            audit={"deterministic": True},
        )
        augmented = augment_stage_edit_repair_review_with_learned_issues(
            deterministic,
            learned,
            session_provenance={"bundle_sha256": "a" * 64},
        )

        self.assertEqual(augmented.decoded_objective, deterministic.decoded_objective)
        self.assertEqual(
            augmented.professional_objective, deterministic.professional_objective
        )
        self.assertEqual(augmented.cadence_objective, deterministic.cadence_objective)
        self.assertEqual(augmented.technical_errors, deterministic.technical_errors)
        self.assertEqual(len(augmented.issue_windows), 2)
        self.assertEqual(
            augmented.audit["learned_reviewer_issue_suggestions"]["authority"],
            "issue_windows_only_deterministic_objectives_unchanged",
        )


if __name__ == "__main__":
    unittest.main()
