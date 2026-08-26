from __future__ import annotations

import json
import unittest

from crosscut.direct_member_look import (
    DirectMemberLookObservation,
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

    def test_observations_must_retain_exact_segment_order(self) -> None:
        observations = (
            DirectMemberLookObservation(1, 0, 1.0, 0.4, 0.0, 0.0, 0.1, 3),
            DirectMemberLookObservation(0, 1, 1.0, 0.4, 0.0, 0.0, 0.1, 3),
        )

        with self.assertRaisesRegex(ValueError, "segment order"):
            solve_direct_member_look_adjustments(observations)


if __name__ == "__main__":
    unittest.main()
