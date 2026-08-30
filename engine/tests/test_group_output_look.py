from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable
import unittest

from crosscut.group_output_look import (
    DecodedLookObjective,
    GroupDecodedLookConfig,
    decoded_look_objective,
    refine_group_render_plan_from_decoded_proxy,
)
from crosscut.ive import AspectRatio
from crosscut.render import (
    RenderPlan,
    RenderSegment,
    SourceLookAdjustment,
    segment_frame_counts,
)
from crosscut.rendered_proxy_review import (
    PERCEPTUAL_SOURCE_ADMISSION_ANALYSIS_BASIS,
    SEGMENT_INTERIOR_GEOMETRY_ANALYSIS_BASIS,
    SEGMENT_INTERIOR_GEOMETRY_CONTRACT,
    SEGMENT_INTERIOR_GEOMETRY_SCHEMA_VERSION,
    _canonical_sha256,
    _render_timeline_binding_payload,
    verified_file_content_sha256,
)


def _plan(directory: Path) -> RenderPlan:
    sources = tuple(directory / name for name in ("a.mp4", "b.mp4", "c.mp4"))
    for index, source in enumerate(sources):
        if not source.exists():
            source.write_bytes(f"source-{index}".encode())
    return RenderPlan(
        sources=sources,
        segments=(
            RenderSegment(0, 0.0, 1.0, 0.0),
            RenderSegment(1, 1.0, 1.0, 1.0),
            RenderSegment(2, 2.0, 1.0, 2.0),
            RenderSegment(0, 3.0, 1.0, 3.0),
        ),
        reference_source=0,
        aspect_ratio=AspectRatio.LANDSCAPE,
        output=directory / "out.mp4",
    )


def _multi_source_plan(directory: Path, source_count: int) -> RenderPlan:
    sources = tuple(directory / f"xoxz-{index}.mp4" for index in range(source_count))
    for index, source in enumerate(sources):
        source.write_bytes(f"xoxz-source-{index}".encode())
    return RenderPlan(
        sources=sources,
        segments=tuple(
            RenderSegment(index, index * 0.5, 0.5, index * 0.5)
            for index in range(source_count)
        ),
        reference_source=0,
        aspect_ratio=AspectRatio.LANDSCAPE,
        output=directory / "xoxz-out.mp4",
    )


def _review(
    plan: RenderPlan,
    *,
    lumas: tuple[float, float, float] = (0.20, 0.50, 0.80),
    details: tuple[float, float, float] = (0.01, 0.02, 0.04),
    chromas: tuple[tuple[float, float], ...] = (
        (-0.10, 0.05),
        (0.0, 0.0),
        (0.10, -0.05),
    ),
    contrasts: tuple[float, ...] = (0.20, 0.30, 0.40),
    penalty: float = 20.0,
    passed: bool = False,
) -> dict[str, object]:
    if not (
        len(plan.sources)
        == len(lumas)
        == len(details)
        == len(chromas)
        == len(contrasts)
    ):
        raise ValueError("test review scalar arrays must match the plan sources")
    rows = []
    for source_index, (luma, detail) in enumerate(zip(lumas, details, strict=True)):
        rows.append(
            {
                "source_index": source_index,
                "decoded_output_frame_count": 30,
                "median_detail": detail,
                "median_luma": luma,
                "median_chroma_a": chromas[source_index][0],
                "median_chroma_b": chromas[source_index][1],
                "median_contrast": contrasts[source_index],
            }
        )
    proxy_identity = {
        "width": 854,
        "height": 480,
        "fps": plan.fps,
        "decoded_frame_count": plan.output_frame_count,
        "duration_seconds": plan.render_duration,
        "render_codec": "libx264",
        "content_sha256": "e" * 64,
    }
    source_content_sha256 = tuple(
        verified_file_content_sha256(path) for path in plan.sources
    )
    geometry_evidence = {
        "schema_version": SEGMENT_INTERIOR_GEOMETRY_SCHEMA_VERSION,
        "analysis_basis": SEGMENT_INTERIOR_GEOMETRY_ANALYSIS_BASIS,
        "contract": SEGMENT_INTERIOR_GEOMETRY_CONTRACT,
        "proxy_identity": proxy_identity,
        "proxy_identity_sha256": _canonical_sha256(proxy_identity),
        "render_timeline_binding_sha256": _canonical_sha256(
            _render_timeline_binding_payload(
                plan,
                segment_frame_counts(plan),
                source_content_sha256=source_content_sha256,
            )
        ),
    }
    geometry_evidence["evidence_sha256"] = _canonical_sha256(geometry_evidence)

    def worst_pair(
        values: tuple[Any, ...], distance: Callable[[Any, Any], float]
    ) -> tuple[list[int], float]:
        measured = [
            (float(distance(left, right)), left_index, right_index)
            for left_index, left in enumerate(values)
            for right_index, right in enumerate(
                values[left_index + 1 :], left_index + 1
            )
        ]
        value, left_index, right_index = max(
            measured,
            key=lambda item: (item[0], -item[1], -item[2]),
        )
        return [left_index, right_index], value

    sharpness_indices, sharpness_value = worst_pair(
        tuple(details),
        lambda left, right: abs(
            math.log((float(right) + 1e-4) / (float(left) + 1e-4))
        ),
    )
    luma_indices, luma_value = worst_pair(
        tuple(lumas),
        lambda left, right: abs(float(right) - float(left)),
    )
    chroma_indices, chroma_value = worst_pair(
        tuple(chromas),
        lambda left, right: math.hypot(
            right[0] - left[0],
            right[1] - left[1],
        ),
    )
    return {
        "proxy": proxy_identity,
        "segment_interior_geometry": geometry_evidence,
        "perceptual_source_admission": {
            "analysis_basis": PERCEPTUAL_SOURCE_ADMISSION_ANALYSIS_BASIS,
            "index_domains": {
                "source_index": "render_plan_sources_array_index",
                "boundary_index": "render_plan_transition_index",
            },
            "source_median_consistency": {
                "per_source": rows,
                "worst_source_pairs": {
                    "sharpness_log": {
                        "source_indices": sharpness_indices,
                        "value": sharpness_value,
                        "maximum_allowed": 0.55,
                    },
                    "luma": {
                        "source_indices": luma_indices,
                        "value": luma_value,
                        "maximum_allowed": 0.18,
                    },
                    "chroma": {
                        "source_indices": chroma_indices,
                        "value": chroma_value,
                        "maximum_allowed": 0.12,
                    },
                },
            },
            "rapid_quality_alternation": {
                "gate_applicable": True,
                "high_quality_shock_fraction": 0.50,
                "maximum_allowed_high_quality_shock_fraction": 0.25,
            },
            "admission": {
                "passed": passed,
                "selection_penalty_points": penalty,
            },
        }
    }


class GroupOutputLookTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = TemporaryDirectory()
        self.directory = Path(self._temporary_directory.name)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def test_decoded_source_medians_create_bounded_same_path_residual(self) -> None:
        plan = _plan(self.directory)
        result = refine_group_render_plan_from_decoded_proxy(
            plan,
            rendered_proxy_review=_review(plan),
            iteration=1,
        )

        self.assertTrue(result.applied)
        self.assertEqual(
            tuple(segment.source_index for segment in result.render_plan.segments),
            tuple(segment.source_index for segment in plan.segments),
        )
        by_source = dict(result.source_residual_adjustments)
        self.assertEqual(by_source[0].brightness, 0.045)
        self.assertEqual(by_source[2].brightness, -0.045)
        self.assertGreater(by_source[0].detail_enhancement, 0.0)
        self.assertEqual(by_source[2].detail_enhancement, 0.0)
        self.assertLessEqual(abs(by_source[0].temperature_balance), 0.090)
        self.assertLessEqual(abs(by_source[0].tint_balance), 0.050)

    def test_second_iteration_composes_but_never_exceeds_renderer_bounds(self) -> None:
        plan = _plan(self.directory)
        first = refine_group_render_plan_from_decoded_proxy(
            plan, rendered_proxy_review=_review(plan), iteration=1
        )
        second = refine_group_render_plan_from_decoded_proxy(
            first.render_plan,
            rendered_proxy_review=_review(first.render_plan),
            iteration=2,
            previous_refinement=first,
        )

        for segment in second.render_plan.segments:
            adjustment = segment.look_adjustment
            assert adjustment is not None
            self.assertLessEqual(abs(adjustment.brightness), 0.18)
            self.assertGreaterEqual(adjustment.contrast, 0.60)
            self.assertLessEqual(adjustment.contrast, 1.40)
            self.assertGreaterEqual(adjustment.saturation, 0.80)
            self.assertLessEqual(adjustment.saturation, 1.20)
            self.assertLessEqual(abs(adjustment.temperature_balance), 0.18)
            self.assertLessEqual(abs(adjustment.tint_balance), 0.10)
            self.assertLessEqual(adjustment.detail_enhancement, 0.90)

    def test_third_iteration_is_rejected(self) -> None:
        plan = _plan(self.directory)
        with self.assertRaisesRegex(ValueError, "outside the fixed bound"):
            refine_group_render_plan_from_decoded_proxy(
                plan, rendered_proxy_review=_review(plan), iteration=3
            )

    def test_incomplete_output_coverage_fails_closed(self) -> None:
        plan = _plan(self.directory)
        review = _review(plan)
        admission = review["perceptual_source_admission"]
        assert isinstance(admission, dict)
        consistency = admission["source_median_consistency"]
        assert isinstance(consistency, dict)
        rows = consistency["per_source"]
        assert isinstance(rows, list)
        assert isinstance(rows[1], dict)
        rows[1]["decoded_output_frame_count"] = 11

        with self.assertRaisesRegex(ValueError, "binding/coverage"):
            refine_group_render_plan_from_decoded_proxy(
                plan, rendered_proxy_review=review, iteration=1
            )

    def test_decoded_evidence_must_bind_the_exact_render_plan(self) -> None:
        plan = _plan(self.directory)
        review = _review(plan)
        changed = replace(
            plan,
            segments=(
                replace(plan.segments[0], source_start=0.125),
                *plan.segments[1:],
            ),
        )

        with self.assertRaisesRegex(ValueError, "exact RenderPlan binding mismatch"):
            refine_group_render_plan_from_decoded_proxy(
                changed,
                rendered_proxy_review=review,
                iteration=1,
            )

    def test_incomplete_objective_scalars_fail_closed(self) -> None:
        plan = _plan(self.directory)
        missing_pair = _review(plan)
        admission = missing_pair["perceptual_source_admission"]
        assert isinstance(admission, dict)
        source = admission["source_median_consistency"]
        assert isinstance(source, dict)
        pairs = source["worst_source_pairs"]
        assert isinstance(pairs, dict)
        pairs["chroma"] = None
        with self.assertRaisesRegex(ValueError, "chroma.*incomplete"):
            decoded_look_objective(missing_pair)

        missing_rapid = _review(plan)
        admission = missing_rapid["perceptual_source_admission"]
        assert isinstance(admission, dict)
        rapid = admission["rapid_quality_alternation"]
        assert isinstance(rapid, dict)
        rapid["high_quality_shock_fraction"] = None
        with self.assertRaisesRegex(ValueError, "rapid evidence is incomplete"):
            decoded_look_objective(missing_rapid)

    def test_second_iteration_requires_cryptographically_bound_lineage(self) -> None:
        plan = _plan(self.directory)
        first = refine_group_render_plan_from_decoded_proxy(
            plan,
            rendered_proxy_review=_review(plan),
            iteration=1,
        )
        with self.assertRaisesRegex(ValueError, "requires exact prior refinement"):
            refine_group_render_plan_from_decoded_proxy(
                first.render_plan,
                rendered_proxy_review=_review(first.render_plan),
                iteration=2,
            )
        with self.assertRaisesRegex(ValueError, "cannot replay lineage"):
            refine_group_render_plan_from_decoded_proxy(
                first.render_plan,
                rendered_proxy_review=_review(first.render_plan),
                iteration=1,
                previous_refinement=first,
            )

        second = refine_group_render_plan_from_decoded_proxy(
            first.render_plan,
            rendered_proxy_review=_review(first.render_plan),
            iteration=2,
            previous_refinement=first,
        )
        self.assertEqual(
            second.lineage.previous_lineage_sha256,
            first.lineage.lineage_sha256,
        )
        self.assertEqual(
            second.lineage.root_render_plan_binding_sha256,
            first.lineage.root_render_plan_binding_sha256,
        )

    def test_xoxz_like_chroma_span_can_close_inside_two_bounded_rounds(self) -> None:
        plan = _multi_source_plan(self.directory, 8)
        initial_lumas = (
            0.5035252869129181,
            0.47652892768383026,
            0.4252817928791046,
            0.2987566590309143,
            0.37742991745471954,
            0.4745262861251831,
            0.3540217876434326,
            0.2671610563993454,
        )
        initial_details = (
            0.2820294350385666,
            0.37975771725177765,
            0.2505289167165756,
            0.22404874861240387,
            0.27727513015270233,
            0.46919724345207214,
            0.35969148576259613,
            0.3081546127796173,
        )
        initial_chromas = (
            (0.09926526993513107, -0.060662757605314255),
            (0.07701405137777328, -0.2790236920118332),
            (0.12377256900072098, -0.07701483368873596),
            (0.11854168772697449, -0.15682454407215118),
            (0.015956368297338486, -0.007015258306637406),
            (0.15369048714637756, -0.08844835311174393),
            (0.2697698026895523, -0.06763644516468048),
            (0.2147771269083023, -0.18508454412221909),
        )
        initial_contrasts = (
            0.7607843137254902,
            0.7588235294117647,
            0.7333333333333333,
            0.6784313725490196,
            0.6313725490196078,
            0.796078431372549,
            0.803921568627451,
            0.6745098039215687,
        )
        first = refine_group_render_plan_from_decoded_proxy(
            plan,
            rendered_proxy_review=_review(
                plan,
                lumas=initial_lumas,
                details=initial_details,
                chromas=initial_chromas,
                contrasts=initial_contrasts,
            ),
            iteration=1,
        )

        def projected(
            chroma: tuple[float, float], adjustment: SourceLookAdjustment
        ) -> tuple[float, float]:
            temperature = adjustment.temperature_balance
            tint = adjustment.tint_balance
            return (
                chroma[0] * adjustment.saturation
                - 0.036559999 * temperature
                - 0.421039015 * tint,
                chroma[1] * adjustment.saturation
                + 0.392331742 * temperature
                + 0.249684229 * tint,
            )

        first_residuals = dict(first.source_residual_adjustments)
        first_chromas = tuple(
            projected(chroma, first_residuals[source_index])
            for source_index, chroma in enumerate(initial_chromas)
        )
        second = refine_group_render_plan_from_decoded_proxy(
            first.render_plan,
            rendered_proxy_review=_review(
                first.render_plan,
                lumas=initial_lumas,
                details=initial_details,
                chromas=first_chromas,
                contrasts=initial_contrasts,
            ),
            iteration=2,
            previous_refinement=first,
        )
        final_by_source = {
            segment.source_index: segment.look_adjustment
            for segment in second.render_plan.segments
        }
        final_chromas = tuple(
            projected(chroma, final_by_source[source_index])
            for source_index, chroma in enumerate(initial_chromas)
            if final_by_source[source_index] is not None
        )
        final_span = max(
            math.hypot(right[0] - left[0], right[1] - left[1])
            for position, left in enumerate(final_chromas)
            for right in final_chromas[position + 1 :]
        )
        self.assertAlmostEqual(
            GroupDecodedLookConfig().maximum_saturation_step,
            0.15,
        )
        self.assertLessEqual(final_span, 0.12)

    def test_objective_requires_strict_scalar_improvement(self) -> None:
        plan = _plan(self.directory)
        before = decoded_look_objective(_review(plan))
        after_review = _review(plan, penalty=4.0, passed=False)
        admission = after_review["perceptual_source_admission"]
        assert isinstance(admission, dict)
        source = admission["source_median_consistency"]
        assert isinstance(source, dict)
        pairs = source["worst_source_pairs"]
        assert isinstance(pairs, dict)
        assert isinstance(pairs["luma"], dict)
        assert isinstance(pairs["chroma"], dict)
        assert isinstance(pairs["sharpness_log"], dict)
        pairs["luma"]["value"] = 0.20
        pairs["chroma"]["value"] = 0.13
        pairs["sharpness_log"]["value"] = 0.60
        rapid = admission["rapid_quality_alternation"]
        assert isinstance(rapid, dict)
        rapid["high_quality_shock_fraction"] = 0.26
        after = decoded_look_objective(after_review)

        self.assertTrue(after.strictly_improves(before))
        self.assertFalse(before.strictly_improves(after))
        self.assertFalse(after.strictly_improves(after))

    def test_passing_admission_dominates_lower_failing_excess(self) -> None:
        failing = DecodedLookObjective(False, 0.0, 0.1)
        passing = DecodedLookObjective(True, 10.0, 0.0)
        self.assertTrue(passing.strictly_improves(failing))

    def test_existing_adjustment_is_composed_not_replaced(self) -> None:
        plan = _plan(self.directory)
        segments = tuple(
            RenderSegment(
                segment.source_index,
                segment.source_start,
                segment.duration,
                segment.master_start,
                look_adjustment=SourceLookAdjustment(brightness=0.03),
            )
            for segment in plan.segments
        )
        plan = RenderPlan(
            sources=plan.sources,
            segments=segments,
            reference_source=plan.reference_source,
            aspect_ratio=plan.aspect_ratio,
            output=plan.output,
        )

        result = refine_group_render_plan_from_decoded_proxy(
            plan, rendered_proxy_review=_review(plan), iteration=1
        )
        first = result.render_plan.segments[0].look_adjustment
        assert first is not None
        self.assertGreater(first.brightness, 0.03)


if __name__ == "__main__":
    unittest.main()
