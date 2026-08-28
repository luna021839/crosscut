from __future__ import annotations

from pathlib import Path
import unittest

from crosscut.direct_member_rendered_continuity import (
    DirectMemberRenderedContinuityConfig,
    DirectMemberRenderedContinuityResult,
)
from crosscut.direct_member_rendered_proxy import (
    DirectMemberRenderedProxyPreflight,
    render_and_review_direct_member_proxy,
)
from crosscut.ive import AspectRatio
from crosscut.render import RenderPlan, RenderSegment, RenderTransition
from crosscut.rendered_proxy_review import (
    RenderedProxyReview,
    derive_proxy_render_plan,
)


def _plan() -> RenderPlan:
    return RenderPlan(
        sources=(Path("one.mp4"), Path("two.mp4")),
        segments=(
            RenderSegment(0, 0.0, 0.4, 0.0),
            RenderSegment(1, 0.4, 0.4, 0.4),
        ),
        reference_source=0,
        aspect_ratio=AspectRatio.PORTRAIT,
        output=Path("full.mp4"),
        width=1080,
        height=1920,
        fps=10.0,
        transitions=(RenderTransition(),),
        member_pose_dissolve_requested=True,
        member_pose_dissolve_effective=False,
        member_pose_dissolve_count=0,
    )


def _continuity(
    *,
    passed: bool,
    indices: tuple[int, ...] = (),
) -> DirectMemberRenderedContinuityResult:
    reason = "rendered_boundary_position_discontinuity"
    return DirectMemberRenderedContinuityResult(
        passed=passed,
        failure_reasons=() if passed else (reason,),
        failure_boundary_indices={} if passed else {reason: indices},
        metrics={},
        boundaries=(),
        config=DirectMemberRenderedContinuityConfig(),
    )


def _generic_payload(*, passed: bool, perceptual_targets=()):
    return {
        "passed": passed,
        "review_policy": {
            "minimum_overall_quality_p10": 0.36,
            "maximum_source_quality_median_span": 0.22,
        },
        "overall_quality": {"p10": 0.80},
        "source_quality": {"median_span": 0.05},
        "introduced_cross_source_boundary_issue_indices": [0],
        "perceptual_source_admission": {
            "admission": {
                "passed": not perceptual_targets,
                "repair_target_boundary_indices": list(perceptual_targets),
            }
        },
        "boundary_domains": {
            "tier_d_continuity": {"passed": True},
            "native_source_cut": {"passed": True},
        },
        "boundaries": [],
        "introduced_cross_source_technical_issue_reasons": [],
        "geometry_provider_failure_count": 0,
    }


class DirectMemberRenderedProxyTests(unittest.TestCase):
    @staticmethod
    def _refinement_result(
        *,
        quality_passed: bool,
        continuity_passed: bool,
        repair_targets: tuple[int, ...],
        position_p95: float,
        weighted_score: float,
    ) -> DirectMemberRenderedProxyPreflight:
        return DirectMemberRenderedProxyPreflight(
            {
                "passed": quality_passed and continuity_passed,
                "member_output_quality_passed": quality_passed,
                "member_output_quality_failure_reasons": (
                    [] if quality_passed else ["rapid_quality_alternation"]
                ),
                "direct_member_rendered_continuity_passed": (
                    continuity_passed
                ),
                "repair_target_boundary_indices": list(repair_targets),
                "rendered_proxy_review": {
                    "weighted_score": weighted_score,
                },
                "rendered_continuity": {
                    "metrics": {
                        "attention_position_mismatch_p95": position_p95,
                    },
                    "thresholds": {
                        "maximum_attention_position_mismatch_p95": 0.12,
                    },
                },
            }
        )

    def test_refinement_rank_rejects_a_later_visual_regression(self) -> None:
        base = self._refinement_result(
            quality_passed=False,
            continuity_passed=False,
            repair_targets=(0,),
            position_p95=0.14,
            weighted_score=62.0,
        )
        regressed = self._refinement_result(
            quality_passed=False,
            continuity_passed=False,
            repair_targets=(0, 1),
            position_p95=0.40,
            weighted_score=64.0,
        )

        self.assertLess(base.refinement_rank(), regressed.refinement_rank())

    def test_refinement_rank_prefers_a_completed_member_quality_gate(self) -> None:
        incomplete = self._refinement_result(
            quality_passed=False,
            continuity_passed=False,
            repair_targets=(0, 1),
            position_p95=0.20,
            weighted_score=61.0,
        )
        quality_fixed = self._refinement_result(
            quality_passed=True,
            continuity_passed=False,
            repair_targets=(0, 1),
            position_p95=0.20,
            weighted_score=61.0,
        )

        self.assertLess(
            quality_fixed.refinement_rank(),
            incomplete.refinement_rank(),
        )

    def test_combines_generic_and_exact_member_gates(self) -> None:
        def render_review(plan, output, **kwargs):
            self.assertEqual(kwargs["video_codec"], "h264_videotoolbox")
            proxy = derive_proxy_render_plan(plan, output)
            return proxy, RenderedProxyReview(_generic_payload(passed=True))

        proxy, result = render_and_review_direct_member_proxy(
            _plan(),
            Path("proxy.mp4"),
            render_review=render_review,
            continuity_evaluator=lambda *_args, **_kwargs: _continuity(
                passed=True
            ),
        )

        self.assertEqual((proxy.width, proxy.height), (480, 854))
        self.assertTrue(result.passed)
        self.assertEqual(result.repair_target_boundary_indices, ())

    def test_unions_perceptual_and_continuity_repair_targets(self) -> None:
        def render_review(plan, output, **_kwargs):
            proxy = derive_proxy_render_plan(plan, output)
            return proxy, RenderedProxyReview(
                _generic_payload(passed=False, perceptual_targets=(0, 2))
            )

        _proxy, result = render_and_review_direct_member_proxy(
            _plan(),
            Path("proxy.mp4"),
            render_review=render_review,
            continuity_evaluator=lambda *_args, **_kwargs: _continuity(
                passed=False,
                indices=(0, 1),
            ),
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.repair_target_boundary_indices, (0, 1, 2))

    def test_generic_group_editorial_issues_do_not_replace_member_gate(self) -> None:
        def render_review(plan, output, **_kwargs):
            proxy = derive_proxy_render_plan(plan, output)
            return proxy, RenderedProxyReview(_generic_payload(passed=False))

        _proxy, result = render_and_review_direct_member_proxy(
            _plan(),
            Path("proxy.mp4"),
            render_review=render_review,
            continuity_evaluator=lambda *_args, **_kwargs: _continuity(
                passed=True
            ),
        )

        self.assertTrue(result.passed)
        self.assertFalse(result.payload["generic_rendered_proxy_passed"])
        self.assertTrue(result.payload["member_output_quality_passed"])

    def test_ranks_worst_continuity_boundaries_for_bounded_reselection(self) -> None:
        def render_review(plan, output, **_kwargs):
            proxy = derive_proxy_render_plan(plan, output)
            return proxy, RenderedProxyReview(_generic_payload(passed=True))

        continuity = DirectMemberRenderedContinuityResult(
            passed=False,
            failure_reasons=(
                "rendered_boundary_position_discontinuity",
                "rendered_boundary_motion_direction_opposed",
            ),
            failure_boundary_indices={
                "rendered_boundary_position_discontinuity": (2,),
                "rendered_boundary_motion_direction_opposed": (1, 2),
            },
            metrics={},
            boundaries=(
                {
                    "boundary_index": 1,
                    "attention_position_mismatch": 0.13,
                    "attention_scale_mismatch": 0.02,
                },
                {
                    "boundary_index": 2,
                    "attention_position_mismatch": 0.30,
                    "attention_scale_mismatch": 0.02,
                },
            ),
            config=DirectMemberRenderedContinuityConfig(),
        )
        _proxy, result = render_and_review_direct_member_proxy(
            _plan(),
            Path("proxy.mp4"),
            render_review=render_review,
            continuity_evaluator=lambda *_args, **_kwargs: continuity,
        )

        self.assertEqual(
            result.ranked_continuity_reselection_boundary_indices(
                maximum_count=2
            ),
            (2, 1),
        )

    def test_quality_only_failure_still_produces_ranked_reselection(self) -> None:
        payload = _generic_payload(
            passed=False,
            perceptual_targets=(3, 4),
        )
        payload["perceptual_source_admission"][
            "rapid_quality_alternation"
        ] = {
            "boundaries": [
                {
                    "boundary_index": 3,
                    "normalized_quality_shock": 1.2,
                },
                {
                    "boundary_index": 4,
                    "normalized_quality_shock": 2.4,
                },
            ]
        }

        def render_review(plan, output, **_kwargs):
            proxy = derive_proxy_render_plan(plan, output)
            return proxy, RenderedProxyReview(payload)

        _proxy, result = render_and_review_direct_member_proxy(
            _plan(),
            Path("proxy.mp4"),
            render_review=render_review,
            continuity_evaluator=lambda *_args, **_kwargs: _continuity(
                passed=True
            ),
        )

        self.assertFalse(result.passed)
        self.assertEqual(
            result.ranked_reselection_boundary_indices(maximum_count=2),
            (4, 3),
        )


if __name__ == "__main__":
    unittest.main()
