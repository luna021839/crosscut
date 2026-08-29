from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2

from crosscut.ive import AspectRatio
from crosscut.render import RenderPlan, RenderSegment, RenderTransition
from crosscut.rendered_candidate_selection import (
    ANALYSIS_BASIS,
    PROFESSIONAL_PLATEAU_ANALYSIS_BASIS,
    PROFESSIONAL_PLATEAU_SELECTION_BASIS,
    PROFESSIONAL_CADENCE_ANALYSIS_BASIS,
    PROFESSIONAL_CADENCE_SELECTION_BASIS,
    RenderedCandidate,
    RenderedCandidateRankMetadata,
    SELECTION_BASIS,
    render_one_candidate_review,
    review_rendered_candidates,
    select_rendered_candidate_reviews,
)
from crosscut.stage_edit_learned_reviewer_adapter import (
    LEARNED_REVIEWER_RERANK_ANALYSIS_BASIS,
    LEARNED_REVIEWER_RERANK_SELECTION_BASIS,
    LearnedReviewerProductionSession,
)
from crosscut.stage_edit_repair_loop import StageEditRepairIssue
from crosscut.professional_reference_cadence import (
    build_mapped_cadence_model,
)
from crosscut.rendered_proxy_review import RenderedProxyReview
from crosscut.professional_reference_gate import (
    DISCONTINUITY_COMPONENTS,
    ProfessionalReferenceCut,
    ProfessionalReferenceProfile,
    ProfessionalReferenceVideo,
)


def _plan(index: int) -> RenderPlan:
    return RenderPlan(
        sources=(Path("source-a.mp4"), Path("source-b.mp4")),
        segments=(
            RenderSegment(0, 0.0, 1.0, 0.0, 1.0),
            RenderSegment(1, 1.0, 1.0, 1.0, 2.0),
        ),
        reference_source=0,
        aspect_ratio=AspectRatio.LANDSCAPE,
        output=Path(f"candidate-{index}.mp4"),
        master_audio_start=0.0,
        width=1920,
        height=1080,
        transitions=(RenderTransition(tier="C", reason="hard"),),
        fps=30.0,
    )


def _payload(*, passed: bool, score: float, issues: float, p10: float) -> dict:
    return {
        "proxy": {"path": "temporary.mp4"},
        "passed": passed,
        "weighted_score": score,
        "boundary_score": {"issue_fraction": issues, "p10": p10},
    }


def _professional_profile(
    root: Path,
    *,
    reference_p10_scores: tuple[float, ...] = (
        50.0,
        51.5,
        53.0,
        54.5,
        56.0,
    ),
) -> ProfessionalReferenceProfile:
    reference_counts = (42, 50, 57, 74, 82)
    if len(reference_p10_scores) != len(reference_counts):
        raise ValueError("test profile needs five score values")
    references = tuple(
        ProfessionalReferenceVideo(
            video_id=f"human-{index}",
            path=root / f"human-{index}.mp4",
            sha256=f"{index + 1:064x}",
            audit_path=root / f"human-{index}-audit.json",
            audit_sha256=f"{index + 11:064x}",
            fps=30.0,
            cuts=tuple(
                ProfessionalReferenceCut(
                    score=reference_p10_scores[index],
                    discontinuities={
                        component: 0.2
                        for component in DISCONTINUITY_COMPONENTS
                    },
                )
                for _ in range(cut_count)
            ),
        )
        for index, cut_count in enumerate(reference_counts)
    )
    return ProfessionalReferenceProfile(
        profile_path=root / "profile.json",
        profile_sha256="a" * 64,
        artist="IVE",
        song="BANG BANG",
        references=references,
        payload={},
    )


def _professional_payload(
    *,
    score: float,
    introduced: int,
    native: int,
    gate_passed: bool = True,
    technical_issue_count: int = 0,
) -> dict:
    return {
        **_payload(
            passed=gate_passed and technical_issue_count == 0,
            score=score,
            issues=0.0,
            p10=score,
        ),
        "introduced_cross_source_count": introduced,
        "native_source_cut_count": native,
        "professional_reference_gate": {
            "passed": gate_passed and technical_issue_count == 0,
            "technical_integrity": {
                "issue_count": technical_issue_count,
                "issue_reasons": (
                    [] if technical_issue_count == 0 else ["black_frame"]
                ),
                "passed": technical_issue_count == 0,
            },
        },
    }


class _SyntheticCadenceProfile:
    def __init__(self) -> None:
        self.model = build_mapped_cadence_model(
            {
                f"human-{index}": (1.4 + jitter,)
                for index, jitter in enumerate(
                    (-0.04, -0.02, 0.0, 0.02, 0.04)
                )
            },
            master_duration_seconds=3.0,
        )

    def score_candidate_boundaries(self, *args, **kwargs):
        return self.model.score_candidate_boundaries(*args, **kwargs)

    def provenance_dict(self) -> dict:
        return {
            "profile_path": "/synthetic/cadence.json",
            "profile_file_sha256": "b" * 64,
            "profile_content_sha256": "c" * 64,
            "artist": "IVE",
            "song": "BANG BANG",
        }


def _cadence_plan(index: int, boundary_seconds: float) -> RenderPlan:
    return RenderPlan(
        sources=(Path("source-a.mp4"), Path("source-b.mp4")),
        segments=(
            RenderSegment(0, 0.0, boundary_seconds, 0.0),
            RenderSegment(
                1,
                boundary_seconds,
                2.0 - boundary_seconds,
                boundary_seconds,
            ),
        ),
        reference_source=0,
        aspect_ratio=AspectRatio.LANDSCAPE,
        output=Path(f"candidate-{index}.mp4"),
        master_audio_start=0.0,
        width=1920,
        height=1080,
        transitions=(RenderTransition(tier="C", reason="hard"),),
        fps=30.0,
    )


class RenderedCandidateSelectionTests(unittest.TestCase):
    @staticmethod
    def _learned_session(
        scores: dict[int, float],
        *,
        issue_indices: dict[int, tuple[int, ...]] | None = None,
        technical_failures: tuple[int, ...] = (),
    ) -> LearnedReviewerProductionSession:
        session = Mock(spec=LearnedReviewerProductionSession)
        session.provenance_dict.return_value = {
            "checkpoint_id": "production-reviewer",
            "bundle_sha256": "a" * 64,
        }

        def review_candidate(*, candidate_index: int, **_kwargs):
            learned = Mock()
            learned.materialization.technical_zero_tolerance_passed = (
                candidate_index not in technical_failures
            )
            learned.prediction.learned_quality_probability = scores[candidate_index]
            indices = (issue_indices or {}).get(candidate_index, ())
            learned.repair_issue_boundary_indices = indices
            learned.repair_issues = tuple(
                StageEditRepairIssue(
                    f"learned-{candidate_index}-{index}",
                    1.0,
                    1.1,
                    ("visual_discontinuity",),
                )
                for index in indices
            )
            learned.to_dict.return_value = {
                "candidate_index": candidate_index,
                "learned_quality_probability": scores[candidate_index],
                "technical_zero_tolerance_passed": (
                    candidate_index not in technical_failures
                ),
            }
            return learned

        session.review_candidate.side_effect = review_candidate
        return session

    def test_learned_reviewer_orders_only_decoded_hard_safe_candidates(self) -> None:
        candidates = tuple(
            RenderedCandidate(index, _plan(index), 100.0 - index)
            for index in range(3)
        )
        reviews = {
            0: _payload(passed=True, score=90.0, issues=0.0, p10=90.0),
            1: _payload(passed=True, score=80.0, issues=0.0, p10=80.0),
            2: _payload(passed=False, score=99.0, issues=0.0, p10=99.0),
        }

        def fake(plan: RenderPlan, _output: Path):
            index = int(plan.output.stem.split("-")[-1])
            return plan, RenderedProxyReview(reviews[index])

        session = self._learned_session(
            {0: 0.20, 1: 0.90, 2: 1.00},
            issue_indices={1: (0,)},
        )
        with tempfile.TemporaryDirectory() as directory:
            result = review_rendered_candidates(
                candidates,
                scratch_directory=Path(directory),
                maximum_review_count=3,
                minimum_required_candidate_count=3,
                render_review=fake,
                learned_reviewer_session=session,
            )

        self.assertEqual(result.selected_candidate_index, 1)
        payload = result.to_dict()
        self.assertEqual(payload["analysis_basis"], LEARNED_REVIEWER_RERANK_ANALYSIS_BASIS)
        self.assertEqual(payload["selection_basis"], LEARNED_REVIEWER_RERANK_SELECTION_BASIS)
        audit = payload["learned_reviewer_rerank"]
        self.assertEqual(audit["deterministic_selected_candidate_index"], 0)
        self.assertEqual(audit["selected_repair_issue_boundary_indices"], [0])
        evidence = {item["candidate_index"]: item for item in audit["candidates"]}
        self.assertFalse(evidence[2]["learned_ordering_eligible"])
        self.assertIn(
            "deterministic_decoded_review_failed",
            evidence[2]["exclusion_reasons"],
        )
        self.assertFalse(audit["policy"]["learned_output_can_override_pass_fail"])

    def test_incremental_reviews_are_ranked_without_rendering_again(self) -> None:
        candidates = tuple(
            RenderedCandidate(index, _plan(index), 100.0 - index)
            for index in range(3)
        )
        payloads = {
            0: _payload(passed=True, score=80.0, issues=0.1, p10=70.0),
            1: _payload(passed=True, score=90.0, issues=0.0, p10=80.0),
            2: _payload(passed=False, score=99.0, issues=0.0, p10=99.0),
        }
        render_calls: list[int] = []

        def fake(plan: RenderPlan, _output: Path):
            index = int(plan.output.stem.split("-")[-1])
            render_calls.append(index)
            return plan, RenderedProxyReview(payloads[index])

        with tempfile.TemporaryDirectory() as directory:
            reviews = tuple(
                render_one_candidate_review(
                    candidate,
                    scratch_directory=Path(directory),
                    render_review=fake,
                )
                for candidate in candidates
            )
            calls_after_render = tuple(render_calls)
            result = select_rendered_candidate_reviews(
                reviews,
                requested_review_count=3,
                minimum_required_candidate_count=3,
            )

        self.assertEqual(calls_after_render, (0, 1, 2))
        self.assertEqual(render_calls, [0, 1, 2])
        self.assertEqual(result.selected_candidate_index, 1)
        self.assertEqual(result.successful_review_count, 3)
        for review in result.reviews:
            assert review.review_payload is not None
            self.assertIsNone(review.review_payload["proxy"]["path"])
            self.assertTrue(
                review.review_payload["proxy"][
                    "temporary_proxy_removed_after_review"
                ]
            )

    def test_learned_reviewer_cannot_expand_professional_active_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = (
                RenderedCandidate(
                    0,
                    _plan(0),
                    90.0,
                    rank_metadata=RenderedCandidateRankMetadata(15.0),
                ),
                RenderedCandidate(
                    1,
                    _plan(1),
                    99.0,
                    rank_metadata=RenderedCandidateRankMetadata(19.0),
                ),
            )
            reviews = {
                0: _professional_payload(score=70.0, introduced=40, native=20),
                1: _professional_payload(score=69.5, introduced=40, native=20),
            }

            def fake(plan: RenderPlan, _output: Path):
                index = int(plan.output.stem.split("-")[-1])
                return plan, RenderedProxyReview(reviews[index])

            result = review_rendered_candidates(
                candidates,
                scratch_directory=root,
                maximum_review_count=2,
                minimum_required_candidate_count=2,
                render_review=fake,
                professional_reference_profile=_professional_profile(root),
                learned_reviewer_session=self._learned_session({0: 0.1, 1: 1.0}),
            )

        self.assertEqual(result.selected_candidate_index, 0)
        audit = result.to_dict()["learned_reviewer_rerank"]
        self.assertEqual(
            audit["policy"]["professional_active_safe_pool"],
            "dynamics_rerank_eligible",
        )
        evidence = {item["candidate_index"]: item for item in audit["candidates"]}
        self.assertFalse(evidence[1]["learned_ordering_eligible"])
        self.assertIn(
            "outside_existing_professional_active_safe_pool",
            evidence[1]["exclusion_reasons"],
        )

    def test_configured_learned_reviewer_fails_closed_on_missing_review(self) -> None:
        candidates = (
            RenderedCandidate(0, _plan(0), 90.0),
            RenderedCandidate(1, _plan(1), 80.0),
        )

        def fake(plan: RenderPlan, _output: Path):
            index = int(plan.output.stem.split("-")[-1])
            if index == 0:
                raise ValueError("synthetic decode failure")
            return plan, RenderedProxyReview(
                _payload(passed=True, score=80.0, issues=0.0, p10=80.0)
            )

        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError,
            "requires every candidate",
        ):
            review_rendered_candidates(
                candidates,
                scratch_directory=Path(directory),
                maximum_review_count=2,
                minimum_required_candidate_count=2,
                render_review=fake,
                learned_reviewer_session=self._learned_session({0: 0.5, 1: 0.5}),
            )

    def test_cadence_prior_is_bounded_inside_existing_safe_dynamics_set(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quality_profile = _professional_profile(root)
            cadence_profile = _SyntheticCadenceProfile()
            boundaries = (0.8, 1.4, 1.4)
            rates = (15.0, 14.0, 17.4)
            candidates = tuple(
                RenderedCandidate(
                    index,
                    _cadence_plan(index, boundary),
                    90.0 - index,
                    rank_metadata=RenderedCandidateRankMetadata(rates[index]),
                )
                for index, boundary in enumerate(boundaries)
            )
            reviews = {
                0: _professional_payload(score=70.0, introduced=1, native=41),
                1: _professional_payload(score=69.5, introduced=1, native=41),
                # Perfect cadence evidence cannot rescue a technical failure.
                2: _professional_payload(
                    score=71.0,
                    introduced=1,
                    native=41,
                    technical_issue_count=1,
                ),
            }

            def fake(plan: RenderPlan, _output: Path):
                index = int(plan.output.stem.split("-")[-1])
                payload = reviews[index]
                boundary_frame = int(round(boundaries[index] * plan.fps))
                payload["proxy"].update(
                    {
                        "fps": plan.fps,
                        "decoded_frame_count": plan.output_frame_count,
                        "duration_seconds": plan.render_duration,
                    }
                )
                payload["boundaries"] = [
                    {
                        "cut_frame_index": boundary_frame,
                        "outgoing_source_index": 0,
                        "incoming_source_index": 1,
                    }
                ]
                return (plan, RenderedProxyReview(payload))

            result = review_rendered_candidates(
                candidates,
                scratch_directory=root,
                maximum_review_count=3,
                minimum_required_candidate_count=3,
                render_review=fake,
                professional_reference_profile=quality_profile,
                professional_reference_cadence_profile=(
                    cadence_profile  # type: ignore[arg-type]
                ),
            )

        self.assertEqual(result.selected_candidate_index, 1)
        serialized = result.to_dict()
        self.assertEqual(
            serialized["analysis_basis"],
            PROFESSIONAL_CADENCE_ANALYSIS_BASIS,
        )
        self.assertEqual(
            serialized["selection_basis"],
            PROFESSIONAL_CADENCE_SELECTION_BASIS,
        )
        cadence = serialized["professional_cadence_prior"]
        self.assertEqual(
            cadence["profile"]["profile_file_sha256"],
            "b" * 64,
        )
        evidence = {
            item["candidate_index"]: item for item in cadence["candidates"]
        }
        self.assertEqual(evidence[0]["raw_prior_0_to_1"], 0.0)
        self.assertGreater(evidence[1]["raw_prior_0_to_1"], 0.9)
        self.assertGreater(evidence[1]["applied_bonus_points"], 0.0)
        self.assertEqual(evidence[2]["applied_bonus_points"], 0.0)
        self.assertFalse(evidence[2]["eligible_for_ranking_bonus"])
        self.assertTrue(
            evidence[1]["clock_validation"][
                "source_switch_clock_matches_render_plan"
            ]
        )

    def test_cadence_profile_fails_closed_on_proxy_clock_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = RenderedCandidate(
                0,
                _cadence_plan(0, 1.4),
                90.0,
                rank_metadata=RenderedCandidateRankMetadata(15.0),
            )

            def fake(plan: RenderPlan, _output: Path):
                payload = _professional_payload(
                    score=70.0,
                    introduced=1,
                    native=41,
                )
                payload["proxy"].update(
                    {
                        "fps": plan.fps,
                        "decoded_frame_count": plan.output_frame_count - 1,
                        "duration_seconds": plan.render_duration,
                    }
                )
                payload["boundaries"] = [
                    {
                        "cut_frame_index": 42,
                        "outgoing_source_index": 0,
                        "incoming_source_index": 1,
                    }
                ]
                return (plan, RenderedProxyReview(payload))

            with self.assertRaisesRegex(ValueError, "frame count"):
                review_rendered_candidates(
                    (candidate,),
                    scratch_directory=root,
                    maximum_review_count=1,
                    minimum_required_candidate_count=1,
                    render_review=fake,
                    professional_reference_profile=_professional_profile(root),
                    professional_reference_cadence_profile=(
                        _SyntheticCadenceProfile()  # type: ignore[arg-type]
                    ),
                )

    def test_cadence_profile_cannot_run_without_professional_quality_gate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "quality profile"):
                review_rendered_candidates(
                    (
                        RenderedCandidate(
                            0,
                            _cadence_plan(0, 1.4),
                            90.0,
                            rank_metadata=RenderedCandidateRankMetadata(15.0),
                        ),
                    ),
                    scratch_directory=Path(directory),
                    maximum_review_count=1,
                    minimum_required_candidate_count=1,
                    professional_reference_cadence_profile=(
                        _SyntheticCadenceProfile()  # type: ignore[arg-type]
                    ),
                )

    def test_professional_plateau_prefers_human_like_higher_switch_density(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = _professional_profile(root)
            rates = (13.6, 17.2, 17.4, 17.4, 18.0, 17.45, 17.3, 15.0)
            pre_render_scores = (98.0, 94.0, 96.0, 99.0, 97.0, 100.0, 99.0, 90.0)
            candidates = tuple(
                RenderedCandidate(
                    index,
                    _plan(index),
                    pre_render_scores[index],
                    rank_metadata=RenderedCandidateRankMetadata(rates[index]),
                )
                for index in range(8)
            )
            reviews = {
                # Best decoded quality remains inside the preferred range.
                0: _professional_payload(score=68.2, introduced=40, native=29),
                # 1.4 points lower: inside the human-derived 1.5-point plateau.
                1: _professional_payload(score=66.8, introduced=51, native=19),
                # Higher dynamics, but just outside the quality plateau.
                2: _professional_payload(score=66.6, introduced=50, native=20),
                # Strong quality/dynamics, but 83 visible cuts exceeds human max 82.
                3: _professional_payload(score=67.9, introduced=60, native=23),
                # Strong quality, but outside the preferred 13.5-17.5 range.
                4: _professional_payload(score=68.0, introduced=50, native=20),
                # Technical failures are never eligible despite a strong score.
                5: _professional_payload(
                    score=68.1,
                    introduced=50,
                    native=20,
                    technical_issue_count=1,
                ),
                # A professional gate failure is likewise ineligible.
                6: _professional_payload(
                    score=69.0,
                    introduced=50,
                    native=20,
                    gate_passed=False,
                ),
                # Inside plateau/rate, but 30 visible cuts is below human min 42.
                7: _professional_payload(score=67.0, introduced=20, native=10),
            }

            def fake(plan: RenderPlan, _output: Path):
                index = int(plan.output.stem.split("-")[-1])
                return (plan, RenderedProxyReview(reviews[index]))

            result = review_rendered_candidates(
                candidates,
                scratch_directory=root,
                maximum_review_count=8,
                minimum_required_candidate_count=8,
                render_review=fake,
                professional_reference_profile=profile,
            )

        self.assertEqual(result.selected_candidate_index, 1)
        serialized = result.to_dict()
        self.assertEqual(
            serialized["analysis_basis"],
            PROFESSIONAL_PLATEAU_ANALYSIS_BASIS,
        )
        self.assertEqual(
            serialized["selection_basis"],
            PROFESSIONAL_PLATEAU_SELECTION_BASIS,
        )
        rerank = serialized["professional_plateau_rerank"]
        self.assertEqual(
            rerank["selection_mode"],
            "preferred_dynamics_within_professional_quality_plateau",
        )
        self.assertEqual(rerank["default_selected_candidate_index"], 0)
        self.assertEqual(rerank["selected_candidate_index"], 1)
        self.assertAlmostEqual(
            rerank["human_envelope"]["reference_score_p10_range"],
            6.0,
        )
        self.assertAlmostEqual(
            rerank["human_envelope"]["plateau_width_points"],
            1.5,
        )
        self.assertEqual(
            rerank["human_envelope"]["reference_visible_cut_counts"],
            [42, 50, 57, 74, 82],
        )
        evidence = {
            item["candidate_index"]: item for item in rerank["candidates"]
        }
        self.assertTrue(evidence[1]["dynamics_rerank_eligible"])
        self.assertFalse(evidence[2]["within_quality_plateau"])
        self.assertFalse(
            evidence[3]["within_human_visible_cut_count_envelope"]
        )
        self.assertFalse(evidence[4]["within_preferred_source_switch_rate"])
        self.assertFalse(evidence[5]["professional_eligible"])
        self.assertEqual(
            evidence[5]["professional_technical_issue_count"],
            1,
        )
        self.assertFalse(evidence[6]["professional_reference_gate_passed"])

    def test_professional_plateau_falls_back_to_quality_when_rates_are_outside(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = _professional_profile(root)
            candidates = (
                RenderedCandidate(
                    0,
                    _plan(0),
                    90.0,
                    rank_metadata=RenderedCandidateRankMetadata(18.0),
                ),
                RenderedCandidate(
                    1,
                    _plan(1),
                    99.0,
                    rank_metadata=RenderedCandidateRankMetadata(19.0),
                ),
            )
            reviews = {
                0: _professional_payload(score=70.0, introduced=40, native=20),
                1: _professional_payload(score=69.5, introduced=40, native=20),
            }

            def fake(plan: RenderPlan, _output: Path):
                index = int(plan.output.stem.split("-")[-1])
                return (plan, RenderedProxyReview(reviews[index]))

            result = review_rendered_candidates(
                candidates,
                scratch_directory=root,
                maximum_review_count=2,
                minimum_required_candidate_count=2,
                render_review=fake,
                professional_reference_profile=profile,
            )

        self.assertEqual(result.selected_candidate_index, 0)
        self.assertEqual(
            result.to_dict()["professional_plateau_rerank"]["selection_mode"],
            (
                "default_quality_rank_within_human_cut_envelope_"
                "no_preferred_dynamics_candidate"
            ),
        )

    def test_professional_plateau_width_has_conservative_one_to_two_point_caps(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = RenderedCandidate(
                0,
                _plan(0),
                90.0,
                rank_metadata=RenderedCandidateRankMetadata(15.0),
            )

            def fake(plan: RenderPlan, _output: Path):
                return (
                    plan,
                    RenderedProxyReview(
                        _professional_payload(
                            score=70.0,
                            introduced=40,
                            native=20,
                        )
                    ),
                )

            narrow = review_rendered_candidates(
                (candidate,),
                scratch_directory=root,
                maximum_review_count=1,
                minimum_required_candidate_count=1,
                render_review=fake,
                professional_reference_profile=_professional_profile(
                    root,
                    reference_p10_scores=(50.0, 50.1, 50.2, 50.3, 50.4),
                ),
            )
            wide = review_rendered_candidates(
                (candidate,),
                scratch_directory=root,
                maximum_review_count=1,
                minimum_required_candidate_count=1,
                render_review=fake,
                professional_reference_profile=_professional_profile(
                    root,
                    reference_p10_scores=(40.0, 45.0, 50.0, 55.0, 60.0),
                ),
            )

        self.assertEqual(
            narrow.to_dict()["professional_plateau_rerank"]["human_envelope"]
            ["plateau_width_points"],
            1.0,
        )
        self.assertEqual(
            wide.to_dict()["professional_plateau_rerank"]["human_envelope"]
            ["plateau_width_points"],
            2.0,
        )

    def test_professional_plateau_requires_human_visible_cut_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = _professional_profile(root)
            candidates = (
                RenderedCandidate(
                    0,
                    _plan(0),
                    99.0,
                    rank_metadata=RenderedCandidateRankMetadata(18.0),
                ),
                RenderedCandidate(
                    1,
                    _plan(1),
                    90.0,
                    rank_metadata=RenderedCandidateRankMetadata(18.0),
                ),
            )
            reviews = {
                0: _professional_payload(score=70.0, introduced=60, native=23),
                1: _professional_payload(score=69.5, introduced=40, native=20),
            }

            def fake(plan: RenderPlan, _output: Path):
                index = int(plan.output.stem.split("-")[-1])
                return (plan, RenderedProxyReview(reviews[index]))

            result = review_rendered_candidates(
                candidates,
                scratch_directory=root,
                maximum_review_count=2,
                minimum_required_candidate_count=2,
                render_review=fake,
                professional_reference_profile=profile,
            )

        self.assertEqual(result.selected_candidate_index, 1)
        audit = result.to_dict()["professional_plateau_rerank"]
        self.assertEqual(
            audit["selection_mode"],
            (
                "default_quality_rank_within_human_cut_envelope_"
                "no_preferred_dynamics_candidate"
            ),
        )
        evidence = {
            item["candidate_index"]: item for item in audit["candidates"]
        }
        self.assertFalse(
            evidence[0]["within_human_visible_cut_count_envelope"]
        )

    def test_professional_plateau_ties_use_weight_pre_render_then_lower_id(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = _professional_profile(root)
            candidates = (
                RenderedCandidate(
                    0,
                    _plan(0),
                    90.0,
                    rank_metadata=RenderedCandidateRankMetadata(17.0),
                ),
                RenderedCandidate(
                    1,
                    _plan(1),
                    91.0,
                    rank_metadata=RenderedCandidateRankMetadata(17.0),
                ),
                RenderedCandidate(
                    2,
                    _plan(2),
                    91.0,
                    rank_metadata=RenderedCandidateRankMetadata(17.0),
                ),
            )
            reviews = {
                index: _professional_payload(
                    score=68.0,
                    introduced=40,
                    native=20,
                )
                for index in range(3)
            }

            def fake(plan: RenderPlan, _output: Path):
                index = int(plan.output.stem.split("-")[-1])
                return (plan, RenderedProxyReview(reviews[index]))

            result = review_rendered_candidates(
                candidates,
                scratch_directory=root,
                maximum_review_count=3,
                minimum_required_candidate_count=3,
                render_review=fake,
                professional_reference_profile=profile,
            )

        self.assertEqual(result.selected_candidate_index, 1)

    def test_professional_dynamics_tie_uses_editorial_ranking_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = tuple(
                RenderedCandidate(
                    index,
                    _plan(index),
                    90.0,
                    rank_metadata=RenderedCandidateRankMetadata(17.0),
                )
                for index in range(2)
            )
            reviews = {
                0: {
                    **_professional_payload(
                        score=80.0,
                        introduced=1,
                        native=41,
                    ),
                    "ranking_weighted_score": 79.0,
                },
                1: {
                    **_professional_payload(
                        score=79.9,
                        introduced=1,
                        native=41,
                    ),
                    "ranking_weighted_score": 80.2,
                },
            }

            def fake(plan: RenderPlan, _output: Path):
                index = int(plan.output.stem.split("-")[-1])
                return (plan, RenderedProxyReview(reviews[index]))

            result = review_rendered_candidates(
                candidates,
                scratch_directory=root,
                maximum_review_count=2,
                minimum_required_candidate_count=2,
                render_review=fake,
                professional_reference_profile=_professional_profile(root),
            )

        self.assertEqual(result.selected_candidate_index, 1)
        self.assertEqual(result.selected.weighted_score, 79.9)
        self.assertEqual(result.selected.ranking_weighted_score, 80.2)

    def test_professional_cadence_tie_uses_editorial_ranking_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cadence_profile = _SyntheticCadenceProfile()
            candidates = tuple(
                RenderedCandidate(
                    index,
                    _cadence_plan(index, 1.4),
                    90.0,
                    rank_metadata=RenderedCandidateRankMetadata(17.0),
                )
                for index in range(2)
            )
            reviews = {
                0: {
                    **_professional_payload(
                        score=80.0,
                        introduced=1,
                        native=41,
                    ),
                    "ranking_weighted_score": 79.0,
                },
                1: {
                    **_professional_payload(
                        score=79.9,
                        introduced=1,
                        native=41,
                    ),
                    "ranking_weighted_score": 80.2,
                },
            }

            def fake(plan: RenderPlan, _output: Path):
                index = int(plan.output.stem.split("-")[-1])
                payload = reviews[index]
                payload["proxy"].update(
                    {
                        "fps": plan.fps,
                        "decoded_frame_count": plan.output_frame_count,
                        "duration_seconds": plan.render_duration,
                    }
                )
                payload["boundaries"] = [
                    {
                        "cut_frame_index": int(round(1.4 * plan.fps)),
                        "outgoing_source_index": 0,
                        "incoming_source_index": 1,
                    }
                ]
                return (plan, RenderedProxyReview(payload))

            result = review_rendered_candidates(
                candidates,
                scratch_directory=root,
                maximum_review_count=2,
                minimum_required_candidate_count=2,
                render_review=fake,
                professional_reference_profile=_professional_profile(root),
                professional_reference_cadence_profile=(
                    cadence_profile  # type: ignore[arg-type]
                ),
            )

        self.assertEqual(result.selected_candidate_index, 1)
        self.assertEqual(result.selected.weighted_score, 79.9)
        self.assertEqual(result.selected.ranking_weighted_score, 80.2)

    def test_no_profile_preserves_legacy_rank_and_schema(self) -> None:
        candidates = (
            RenderedCandidate(
                0,
                _plan(0),
                90.0,
                rank_metadata=RenderedCandidateRankMetadata(13.6),
            ),
            RenderedCandidate(
                1,
                _plan(1),
                99.0,
                rank_metadata=RenderedCandidateRankMetadata(17.2),
            ),
        )
        reviews = {
            0: _payload(passed=True, score=70.0, issues=0.1, p10=70.0),
            1: _payload(passed=True, score=69.0, issues=0.0, p10=80.0),
        }

        def fake(plan: RenderPlan, _output: Path):
            index = int(plan.output.stem.split("-")[-1])
            return (plan, RenderedProxyReview(reviews[index]))

        with tempfile.TemporaryDirectory() as directory:
            result = review_rendered_candidates(
                candidates,
                scratch_directory=Path(directory),
                maximum_review_count=2,
                minimum_required_candidate_count=2,
                render_review=fake,
            )

        self.assertEqual(result.selected_candidate_index, 0)
        serialized = result.to_dict()
        self.assertEqual(serialized["analysis_basis"], ANALYSIS_BASIS)
        self.assertEqual(serialized["selection_basis"], SELECTION_BASIS)
        self.assertNotIn("professional_plateau_rerank", serialized)

    def test_separate_editorial_ranking_score_does_not_mutate_technical_score(
        self,
    ) -> None:
        candidates = (
            RenderedCandidate(0, _plan(0), 90.0),
            RenderedCandidate(1, _plan(1), 90.0),
        )
        reviews = {
            0: {
                **_payload(
                    passed=True,
                    score=80.0,
                    issues=0.0,
                    p10=80.0,
                ),
                "ranking_weighted_score": 79.0,
            },
            1: {
                **_payload(
                    passed=True,
                    score=79.5,
                    issues=0.0,
                    p10=79.5,
                ),
                "ranking_weighted_score": 80.5,
            },
        }

        def fake(plan: RenderPlan, _output: Path):
            index = int(plan.output.stem.split("-")[-1])
            return (plan, RenderedProxyReview(reviews[index]))

        with tempfile.TemporaryDirectory() as directory:
            result = review_rendered_candidates(
                candidates,
                scratch_directory=Path(directory),
                maximum_review_count=2,
                minimum_required_candidate_count=2,
                render_review=fake,
            )

        self.assertEqual(result.selected_candidate_index, 1)
        self.assertEqual(result.selected.weighted_score, 79.5)
        self.assertEqual(result.selected.ranking_weighted_score, 80.5)
        selected = next(
            item for item in result.to_dict()["candidates"] if item["selected"]
        )
        self.assertEqual(selected["weighted_score"], 79.5)
        self.assertEqual(selected["ranking_weighted_score"], 80.5)

    def test_rank_metadata_is_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            RenderedCandidateRankMetadata(float("nan"))
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            RenderedCandidateRankMetadata(-0.1)

    def test_schema_v3_ranking_prefers_visible_cross_source_summary(self) -> None:
        candidates = tuple(
            RenderedCandidate(index, _plan(index), 90.0 - index)
            for index in range(2)
        )
        reviews = {
            0: {
                **_payload(passed=False, score=80.0, issues=0.0, p10=99.0),
                "visible_boundary_score": {
                    "issue_fraction": 1.0,
                    "p10": 40.0,
                },
            },
            1: {
                **_payload(passed=False, score=80.0, issues=1.0, p10=20.0),
                "visible_boundary_score": {
                    "issue_fraction": 0.25,
                    "p10": 70.0,
                },
            },
        }

        def fake(plan: RenderPlan, _output: Path):
            index = int(plan.output.stem.split("-")[-1])
            return (plan, RenderedProxyReview(reviews[index]))

        with tempfile.TemporaryDirectory() as directory:
            result = review_rendered_candidates(
                candidates,
                scratch_directory=Path(directory),
                maximum_review_count=2,
                minimum_required_candidate_count=2,
                render_review=fake,
            )

        self.assertEqual(result.selected_candidate_index, 1)
        serialized = result.to_dict()
        self.assertEqual(serialized["candidates"][1]["issue_fraction"], 0.25)
        self.assertEqual(serialized["candidates"][1]["boundary_score_p10"], 70.0)

    def test_passing_candidate_beats_higher_scoring_failure_and_proxies_are_sequential(self) -> None:
        candidates = tuple(
            RenderedCandidate(index, _plan(index), 90.0 - index)
            for index in range(3)
        )
        reviews = {
            0: _payload(passed=False, score=95.0, issues=0.40, p10=70.0),
            1: _payload(passed=True, score=72.0, issues=0.20, p10=62.0),
            2: _payload(passed=True, score=70.0, issues=0.10, p10=66.0),
        }
        active_counts: list[int] = []
        temporary_roots: list[Path] = []

        def fake(plan: RenderPlan, output: Path):
            if temporary_roots:
                self.assertFalse(temporary_roots[-1].exists())
            temporary_roots.append(output.parent)
            active_counts.append(len(tuple(output.parent.glob("*.mp4"))))
            output.write_bytes(b"proxy")
            index = int(plan.output.stem.split("-")[-1])
            return (plan, RenderedProxyReview(reviews[index]))

        with tempfile.TemporaryDirectory() as directory:
            result = review_rendered_candidates(
                candidates,
                scratch_directory=Path(directory),
                maximum_review_count=3,
                minimum_required_candidate_count=3,
                render_review=fake,
            )
            self.assertFalse(any(Path(directory).rglob("*.mp4")))

        self.assertEqual(active_counts, [0, 0, 0])
        self.assertEqual(len(set(temporary_roots)), 3)
        self.assertFalse(any(root.exists() for root in temporary_roots))
        self.assertEqual(result.selected_candidate_index, 1)
        self.assertTrue(result.passed)
        self.assertEqual(result.passing_candidate_count, 2)
        self.assertTrue(result.minimum_candidate_count_satisfied)
        payload = result.to_dict()
        self.assertTrue(payload["storage_policy"]["one_proxy_at_a_time"])
        self.assertIsNone(
            payload["candidates"][1]["review"]["proxy"]["path"]
        )
        self.assertTrue(
            payload["candidates"][1]["review"]["proxy"][
                "temporary_proxy_removed_after_review"
            ]
        )

    def test_review_error_does_not_abort_remaining_candidates(self) -> None:
        candidates = tuple(
            RenderedCandidate(index, _plan(index), float(index)) for index in range(2)
        )

        def fake(plan: RenderPlan, _output: Path):
            index = int(plan.output.stem.split("-")[-1])
            if index == 0:
                raise ValueError("broken candidate")
            return (
                plan,
                RenderedProxyReview(
                    _payload(passed=False, score=61.0, issues=0.36, p10=59.0)
                ),
            )

        with tempfile.TemporaryDirectory() as directory:
            result = review_rendered_candidates(
                candidates,
                scratch_directory=Path(directory),
                maximum_review_count=2,
                minimum_required_candidate_count=8,
                render_review=fake,
            )

        self.assertEqual(result.successful_review_count, 1)
        self.assertEqual(result.selected_candidate_index, 1)
        self.assertFalse(result.passed)
        self.assertFalse(result.minimum_candidate_count_satisfied)
        self.assertIn("broken candidate", result.reviews[0].error or "")

    def test_opencv_review_error_is_recorded_and_remaining_candidate_runs(self) -> None:
        candidates = tuple(
            RenderedCandidate(index, _plan(index), float(index)) for index in range(2)
        )
        calls: list[int] = []
        temporary_roots: list[Path] = []

        def fake(plan: RenderPlan, output: Path):
            index = int(plan.output.stem.split("-")[-1])
            calls.append(index)
            if temporary_roots:
                self.assertFalse(temporary_roots[-1].exists())
            temporary_roots.append(output.parent)
            if index == 0:
                raise cv2.error("synthetic OpenCV decode failure")
            return (
                plan,
                RenderedProxyReview(
                    _payload(passed=True, score=81.0, issues=0.0, p10=81.0)
                ),
            )

        with tempfile.TemporaryDirectory() as directory:
            result = review_rendered_candidates(
                candidates,
                scratch_directory=Path(directory),
                maximum_review_count=2,
                minimum_required_candidate_count=2,
                render_review=fake,
            )

        self.assertEqual(calls, [0, 1])
        self.assertEqual(result.successful_review_count, 1)
        self.assertEqual(result.selected_candidate_index, 1)
        self.assertTrue(result.passed)
        self.assertTrue(result.minimum_candidate_count_satisfied)
        self.assertIn("cv2.error", result.reviews[0].error or "")
        self.assertIn("OpenCV decode failure", result.reviews[0].error or "")
        self.assertFalse(any(root.exists() for root in temporary_roots))

    def test_duplicate_candidate_indices_are_rejected_before_render(self) -> None:
        duplicate = (
            RenderedCandidate(0, _plan(0), 1.0),
            RenderedCandidate(0, _plan(1), 2.0),
        )
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError, "unique"
        ):
            review_rendered_candidates(
                duplicate,
                scratch_directory=Path(directory),
            )

    def test_proxy_cleanup_failure_stops_before_the_next_candidate(self) -> None:
        candidates = tuple(
            RenderedCandidate(index, _plan(index), float(index))
            for index in range(2)
        )
        calls: list[int] = []

        def fake(plan: RenderPlan, output: Path):
            index = int(plan.output.stem.split("-")[-1])
            calls.append(index)
            output.write_bytes(b"proxy")
            return (
                plan,
                RenderedProxyReview(
                    _payload(passed=True, score=80.0, issues=0.0, p10=80.0)
                ),
            )

        original_unlink = Path.unlink

        def fail_proxy_unlink(path: Path, *args, **kwargs):
            if path.name == "candidate-00.mp4":
                raise OSError("synthetic cleanup failure")
            return original_unlink(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.object(
            Path,
            "unlink",
            new=fail_proxy_unlink,
        ), self.assertRaisesRegex(OSError, "cleanup failure"):
            review_rendered_candidates(
                candidates,
                scratch_directory=Path(directory),
                maximum_review_count=2,
                minimum_required_candidate_count=2,
                render_review=fake,
            )

        self.assertEqual(calls, [0])


if __name__ == "__main__":
    unittest.main()
