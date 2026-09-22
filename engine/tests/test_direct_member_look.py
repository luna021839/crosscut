from __future__ import annotations

import json
import unittest

from crosscut.direct_member_look import (
    DirectMemberLookObservation,
    refine_direct_member_look_plan_from_rendered_proxy,
    solve_direct_member_look_adjustments,
)


class DirectMemberLookTests(unittest.TestCase):
    def test_graph_calibration_reduces_rapid_camera_quality_oscillation(self) -> None:
        sources = (0, 1, 0, 1, 2, 2, 0, 1, 0, 1)
        observations = tuple(
            DirectMemberLookObservation(
                segment_index=index,
                source_index=source,
                duration_seconds=1.5,
                luma=(0.40 if source == 0 else 0.50 if source == 1 else 0.45),
                chroma_a=(0.12 if source == 0 else 0.04 if source == 1 else 0.08),
                chroma_b=(-0.11 if source == 0 else -0.17 if source == 1 else -0.14),
                detail_proxy=(0.13 if source == 0 else 0.16 if source == 1 else 0.145),
                sample_count=6,
            )
            for index, source in enumerate(sources)
        )

        plan = solve_direct_member_look_adjustments(observations)

        self.assertGreaterEqual(plan.rapid_boundary_count, 4)
        self.assertIsNotNone(plan.predicted_high_shock_fraction)
        self.assertLessEqual(plan.predicted_high_shock_fraction, 0.25)
        self.assertTrue(
            any(abs(item.tint_balance) > 1e-6 for item in plan.segment_adjustments)
        )
        # Consecutive pieces of one retained camera run cannot reset the grade.
        self.assertEqual(plan.segment_adjustments[4], plan.segment_adjustments[5])
        payload = plan.to_dict()
        self.assertFalse(payload["privacy"]["raw_frames_persisted"])
        self.assertFalse(
            payload["calibration"]["full_song_proxy_rendered"]
        )
        self.assertEqual(
            payload["calibration"]["color_axes"],
            "ffmpeg_native_yuv_luma_exact_temperature_tint_geq_v1",
        )
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
        self.assertTrue(
            all(
                type(item["above_threshold"]) is bool
                for item in payload["diagnostics"]
            )
        )

    def test_rapid_global_stage_chroma_is_diagnostic_not_a_skin_grade_gate(self) -> None:
        stage_chroma = (0.10, -0.10, -0.10, 0.10) * 3
        observations = tuple(
            DirectMemberLookObservation(
                segment_index=index,
                source_index=index % 2,
                duration_seconds=1.0,
                luma=0.45,
                chroma_a=stage_chroma[index],
                chroma_b=-stage_chroma[index],
                detail_proxy=0.14,
                sample_count=6,
            )
            for index in range(len(stage_chroma))
        )

        plan = solve_direct_member_look_adjustments(observations)

        self.assertEqual(plan.predicted_high_shock_count, 0)
        self.assertEqual(plan.predicted_high_shock_fraction, 0.0)
        self.assertTrue(
            any(
                item["normalized_axes"]["chroma"] > 1.0
                for item in plan.diagnostics
            )
        )

    def test_sparse_source_spans_defer_global_luma_range_to_decoded_proxy(self) -> None:
        observations = tuple(
            DirectMemberLookObservation(
                segment_index=index,
                source_index=index,
                duration_seconds=4.0,
                luma=0.08 + index * 0.16,
                chroma_a=0.02,
                chroma_b=-0.02,
                detail_proxy=0.14,
                sample_count=6,
            )
            for index in range(6)
        )

        plan = solve_direct_member_look_adjustments(observations)

        self.assertFalse(plan.source_span_policy_enforced)
        self.assertEqual(plan.source_span_repeated_source_fraction, 0.0)
        self.assertGreater(
            dict(plan.predicted_source_spans)["luma"],
            plan.config.source_luma_span,
        )
        self.assertTrue(
            plan.to_dict()["source_span_policy"][
                "underidentified_initial_cache_requires_decoded_proxy_qa"
            ]
        )

    def test_repeated_source_spans_retain_strict_global_luma_gate(self) -> None:
        observations = tuple(
            DirectMemberLookObservation(
                segment_index=index,
                source_index=index % 2,
                duration_seconds=4.0,
                luma=0.05 if index % 2 == 0 else 0.95,
                chroma_a=0.02,
                chroma_b=-0.02,
                detail_proxy=0.14,
                sample_count=6,
            )
            for index in range(6)
        )

        with self.assertRaisesRegex(ValueError, "source-span"):
            solve_direct_member_look_adjustments(observations)

    def test_observations_must_retain_exact_segment_order(self) -> None:
        observations = (
            DirectMemberLookObservation(1, 0, 1.0, 0.4, 0.0, 0.0, 0.1, 3),
            DirectMemberLookObservation(0, 1, 1.0, 0.4, 0.0, 0.0, 0.1, 3),
        )

        with self.assertRaisesRegex(ValueError, "segment order"):
            solve_direct_member_look_adjustments(observations)

    def test_encoded_proxy_scalars_produce_one_composed_residual_pass(self) -> None:
        observations = tuple(
            DirectMemberLookObservation(
                segment_index=index,
                source_index=index % 2,
                duration_seconds=1.5,
                luma=0.42 if index % 2 == 0 else 0.50,
                chroma_a=0.10 if index % 2 == 0 else 0.03,
                chroma_b=-0.08 if index % 2 == 0 else -0.16,
                detail_proxy=0.13 if index % 2 == 0 else 0.17,
                sample_count=6,
            )
            for index in range(6)
        )
        base = solve_direct_member_look_adjustments(observations)
        per_run = [
            {
                "run_index": index,
                "source_index": 4 if index % 2 == 0 else 7,
                "duration_seconds": 1.5,
                "decoded_output_frame_count": 45,
                "incoming_boundary_index": None if index == 0 else index - 1,
                "scalars": {
                    "perceptual_detail": 0.10 if index % 2 == 0 else 0.20,
                    "perceptual_luma": 0.36 if index % 2 == 0 else 0.54,
                    "perceptual_chroma_a": 0.16 if index % 2 == 0 else 0.01,
                    "perceptual_chroma_b": -0.04 if index % 2 == 0 else -0.20,
                    "perceptual_contrast": 0.70,
                },
            }
            for index in range(6)
        ]

        refined = refine_direct_member_look_plan_from_rendered_proxy(
            base,
            rendered_proxy_review={
                "perceptual_source_admission": {
                    "rapid_quality_alternation": {"per_run": per_run}
                }
            },
            eligible_request_source_indices=(4, 7),
        )

        self.assertTrue(refined.output_proxy_refinement_applied)
        self.assertEqual(refined.output_proxy_refinement_iteration, 1)
        self.assertTrue(refined.config.gate_rapid_chroma)
        self.assertTrue(refined.config.enable_detail_enhancement)
        self.assertTrue(
            any(
                adjustment.detail_enhancement > 0.0
                for adjustment in refined.segment_adjustments
            )
        )
        self.assertNotEqual(
            refined.segment_adjustments,
            base.segment_adjustments,
        )
        payload = refined.to_dict()
        self.assertTrue(payload["calibration"]["full_song_proxy_rendered"])
        self.assertTrue(
            payload["calibration"][
                "output_domain_residual_composed_with_cached_grade"
            ]
        )


if __name__ == "__main__":
    unittest.main()
