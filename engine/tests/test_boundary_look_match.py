from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from crosscut.boundary_look_match import (
    BoundaryLookMatchConfig,
    analyze_boundary_look_match,
    verify_boundary_look_match_render_domain,
)
from crosscut.ive import AspectRatio
from crosscut.render import (
    RenderPlan,
    RenderSegment,
    SourceLookAdjustment,
)
from crosscut.source_look_match import LookMetricVector


class _MetricCache:
    def __init__(
        self,
        samples: dict[str, tuple[LookMetricVector | None, ...]],
    ) -> None:
        self.samples = samples
        self.decoded_frame_count = 0
        self.cache_hit_count = 0
        self.calls: list[str] = []

    def measure_many(self, path: Path, times_seconds):
        values = tuple(sorted(set(float(value) for value in times_seconds)))
        self.calls.append(path.name)
        samples = self.samples[path.name]
        result = {}
        for index, value in enumerate(values):
            metric = samples[index % len(samples)]
            result[int(round(value * 1_000_000.0))] = metric
            if metric is not None:
                self.decoded_frame_count += 1
        return result


def _metric(
    luma: float,
    contrast: float,
    saturation: float,
    temperature: float = 0.0,
) -> LookMetricVector:
    return LookMetricVector(luma, contrast, saturation, temperature)


def _plan(
    *,
    left_look: SourceLookAdjustment | None = None,
    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE,
) -> RenderPlan:
    width, height = (
        (1920, 1080)
        if aspect_ratio is AspectRatio.LANDSCAPE
        else (1080, 1920)
    )
    return RenderPlan(
        sources=(Path("left.mp4"), Path("right.mp4")),
        segments=(
            RenderSegment(0, 0.0, 1.0, 0.0, look_adjustment=left_look),
            RenderSegment(1, 0.0, 1.0, 1.0),
        ),
        reference_source=0,
        aspect_ratio=aspect_ratio,
        output=Path("out.mp4"),
        width=width,
        height=height,
        fps=30.0,
    )


class BoundaryLookMatchTests(unittest.TestCase):
    def test_stable_residual_becomes_short_post_global_ramps(self) -> None:
        samples = {
            "left.mp4": tuple(_metric(0.38, 0.30, 0.31) for _ in range(7)),
            "right.mp4": tuple(_metric(0.46, 0.35, 0.40) for _ in range(7)),
        }
        cache = _MetricCache(samples)
        # The left source is already brightened globally.  The local residual
        # must use 0.40 vs 0.46, not grade the raw 0.38 difference a second time.
        plan = _plan(left_look=SourceLookAdjustment(brightness=0.02))

        result = analyze_boundary_look_match(
            plan,
            group_16x9_safe=True,
            measurement_cache=cache,  # type: ignore[arg-type]
        )

        self.assertTrue(result.enabled)
        self.assertEqual(result.applied_count, 1)
        decision = result.decisions[0]
        self.assertTrue(decision.applied)
        self.assertEqual(
            decision.reason,
            "stable_bounded_synchronized_boundary_residual",
        )
        self.assertAlmostEqual(
            decision.outgoing_adjustment.brightness,
            0.0252,
            places=4,
        )
        self.assertAlmostEqual(
            decision.incoming_adjustment.brightness,
            -0.0252,
            places=4,
        )
        self.assertLess(
            decision.predicted_distance_after,
            decision.predicted_distance_before,
        )
        adjusted = result.apply(plan)
        left_track = adjusted.segments[0].boundary_look_track
        right_track = adjusted.segments[1].boundary_look_track
        self.assertIsNotNone(left_track)
        self.assertIsNotNone(right_track)
        assert left_track is not None and right_track is not None
        self.assertTrue(left_track.first_keyframe.is_identity)
        self.assertFalse(left_track.final_keyframe.is_identity)
        self.assertFalse(right_track.first_keyframe.is_identity)
        self.assertTrue(right_track.final_keyframe.is_identity)
        payload = result.to_dict()
        self.assertFalse(payload["proxy_thresholds_modified"])
        self.assertFalse(payload["raw_frames_persisted"])
        self.assertTrue(
            payload["static_source_look_accounted_for_before_residual"]
        )
        self.assertTrue(payload["same_canonical_time_pairing_used"])
        self.assertTrue(payload["temperature_residual_never_modified"])
        self.assertIsNotNone(
            payload["decisions"][0]["synchronized_residual_window"]
        )
        repeated = analyze_boundary_look_match(
            plan,
            group_16x9_safe=True,
            measurement_cache=_MetricCache(samples),  # type: ignore[arg-type]
        )
        self.assertEqual(repeated.decisions, result.decisions)
        self.assertEqual(repeated.segment_tracks, result.segment_tracks)

    def test_volatile_or_temperature_dominated_windows_fail_closed(self) -> None:
        volatile = tuple(
            _metric(0.36 if index % 2 else 0.52, 0.32, 0.36)
            for index in range(7)
        )
        stable = tuple(_metric(0.45, 0.34, 0.38) for _ in range(7))
        result = analyze_boundary_look_match(
            _plan(),
            group_16x9_safe=True,
            measurement_cache=_MetricCache(  # type: ignore[arg-type]
                {"left.mp4": volatile, "right.mp4": stable}
            ),
        )
        self.assertEqual(result.applied_count, 0)
        self.assertEqual(
            result.decisions[0].reason,
            "volatile_synchronized_camera_residual",
        )
        self.assertTrue(all(track is None for track in result.segment_tracks))

        temperature = analyze_boundary_look_match(
            _plan(),
            group_16x9_safe=True,
            measurement_cache=_MetricCache(  # type: ignore[arg-type]
                {
                    "left.mp4": tuple(
                        _metric(0.42, 0.34, 0.38, -0.11) for _ in range(7)
                    ),
                    "right.mp4": tuple(
                        _metric(0.45, 0.34, 0.38, 0.11) for _ in range(7)
                    ),
                }
            ),
        )
        self.assertEqual(temperature.applied_count, 1)
        self.assertEqual(
            temperature.decisions[0].reason,
            "stable_bounded_partial_residual_with_yuv_chroma_affine_temperature",
        )
        self.assertGreater(
            temperature.decisions[0].outgoing_adjustment.temperature_balance,
            0.0,
        )
        self.assertLess(
            temperature.decisions[0].incoming_adjustment.temperature_balance,
            0.0,
        )
        self.assertTrue(temperature.to_dict()["temperature_ramps_used"])
        self.assertTrue(
            temperature.to_dict()["neutral_temperature_frames_sample_exact_yuv"]
        )
        self.assertFalse(
            temperature.to_dict()["temperature_residual_never_modified"]
        )

    def test_missing_samples_and_unsafe_aspect_never_apply(self) -> None:
        with self.assertRaisesRegex(ValueError, "chroma-affine bound"):
            BoundaryLookMatchConfig(maximum_temperature_balance=0.020001)
        missing_cache = _MetricCache(
            {
                "left.mp4": (None,) * 7,
                "right.mp4": tuple(_metric(0.45, 0.34, 0.38) for _ in range(7)),
            }
        )
        missing = analyze_boundary_look_match(
            _plan(),
            group_16x9_safe=True,
            measurement_cache=missing_cache,  # type: ignore[arg-type]
        )
        self.assertEqual(
            missing.decisions[0].reason,
            "insufficient_synchronized_boundary_pairs",
        )
        unsafe_cache = _MetricCache({})
        unsafe = analyze_boundary_look_match(
            _plan(aspect_ratio=AspectRatio.PORTRAIT),
            group_16x9_safe=False,
            measurement_cache=unsafe_cache,  # type: ignore[arg-type]
        )
        self.assertFalse(unsafe.enabled)
        self.assertEqual(unsafe.blocked_reason, "output_or_source_aspect_unsafe")
        self.assertEqual(unsafe_cache.calls, [])

    def test_short_segment_preserves_two_frame_endpoint_holds(self) -> None:
        plan = RenderPlan(
            sources=(Path("left.mp4"), Path("right.mp4"), Path("third.mp4")),
            segments=(
                RenderSegment(0, 0.0, 0.3, 0.0),
                RenderSegment(1, 0.0, 0.3, 0.3),
                RenderSegment(2, 0.0, 0.3, 0.6),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=30.0,
        )
        cache = _MetricCache(
            {
                "left.mp4": tuple(_metric(0.39, 0.33, 0.34) for _ in range(7)),
                "right.mp4": tuple(_metric(0.44, 0.35, 0.39) for _ in range(7)),
                "third.mp4": tuple(_metric(0.48, 0.37, 0.43) for _ in range(7)),
            }
        )
        result = analyze_boundary_look_match(
            plan,
            group_16x9_safe=True,
            measurement_cache=cache,  # type: ignore[arg-type]
            config=BoundaryLookMatchConfig(ramp_frame_count=7),
        )
        middle = result.segment_tracks[1]
        self.assertIsNotNone(middle)
        assert middle is not None
        self.assertEqual(len(middle.keyframes), 4)
        self.assertFalse(middle.first_keyframe.is_identity)
        self.assertFalse(middle.final_keyframe.is_identity)
        self.assertEqual(
            (
                middle.keyframes[0].brightness,
                middle.keyframes[0].contrast,
                middle.keyframes[0].saturation,
                middle.keyframes[0].temperature_balance,
            ),
            (
                middle.keyframes[1].brightness,
                middle.keyframes[1].contrast,
                middle.keyframes[1].saturation,
                middle.keyframes[1].temperature_balance,
            ),
        )
        self.assertEqual(
            middle.keyframes[-2].brightness,
            middle.keyframes[-1].brightness,
        )

    def test_synchronized_residual_cancels_shared_flash_but_not_one_sided_flash(self) -> None:
        shared_left = tuple(
            _metric(0.30 if index % 2 else 0.58, 0.34, 0.36)
            for index in range(7)
        )
        shared_right = tuple(
            _metric(0.38 if index % 2 else 0.66, 0.34, 0.36)
            for index in range(7)
        )
        shared = analyze_boundary_look_match(
            _plan(),
            group_16x9_safe=True,
            measurement_cache=_MetricCache(  # type: ignore[arg-type]
                {"left.mp4": shared_left, "right.mp4": shared_right}
            ),
        )
        self.assertEqual(shared.applied_count, 1)
        decision = shared.decisions[0]
        self.assertEqual(
            decision.reason,
            "stable_bounded_synchronized_boundary_residual",
        )
        self.assertGreater(
            decision.outgoing_window.robust_half_range.luma,  # type: ignore[union-attr]
            0.10,
        )
        self.assertLess(
            decision.synchronized_residual_window.mad.luma_delta,  # type: ignore[union-attr]
            1e-6,
        )

        one_sided = analyze_boundary_look_match(
            _plan(),
            group_16x9_safe=True,
            measurement_cache=_MetricCache(  # type: ignore[arg-type]
                {
                    "left.mp4": shared_left,
                    "right.mp4": tuple(
                        _metric(0.44, 0.34, 0.36) for _ in range(7)
                    ),
                }
            ),
        )
        self.assertEqual(one_sided.applied_count, 0)
        self.assertEqual(
            one_sided.decisions[0].reason,
            "volatile_synchronized_camera_residual",
        )

    def test_render_domain_admission_keeps_only_component_safe_improvement(self) -> None:
        samples = {
            "left.mp4": tuple(_metric(0.38, 0.30, 0.31) for _ in range(7)),
            "right.mp4": tuple(_metric(0.46, 0.35, 0.40) for _ in range(7)),
        }
        plan = _plan()
        proposed = analyze_boundary_look_match(
            plan,
            group_16x9_safe=True,
            measurement_cache=_MetricCache(samples),  # type: ignore[arg-type]
        )

        def rendered_review(render_plan, _cut_frame_index):
            adjusted = any(
                segment.boundary_look_track is not None
                for segment in render_plan.segments
            )
            sharpness, exposure, color = (
                (0.12, 0.08, 0.30) if not adjusted else (0.10, 0.05, 0.24)
            )
            return {
                "discontinuities": {
                    "sharpness_discontinuity": sharpness,
                    "exposure_discontinuity": exposure,
                    "color_shock": color,
                    "structure_shock": 0.30,
                    "motion_direction_discontinuity": 0.20,
                },
                "frame_metrics": [
                    {
                        "offset": offset,
                        "mean_luma": 0.40 if offset < 0 else 0.44,
                        "clipped_pixel_ratio": 0.002,
                    }
                    for offset in (-2, -1, 0, 1)
                ],
                "technical_integrity_issue_reasons": [],
                "issue_reasons": [] if adjusted else ["color_shock"],
            }

        admitted = verify_boundary_look_match_render_domain(
            plan,
            proposed,
            review_function=rendered_review,
        )
        self.assertEqual(admitted.applied_count, 1)
        self.assertEqual(admitted.render_verified_count, 1)
        self.assertEqual(admitted.render_rejected_count, 0)
        self.assertTrue(
            admitted.decisions[0].render_verification.passed  # type: ignore[union-attr]
        )
        audit = admitted.to_dict()["render_domain_admission"]
        self.assertTrue(audit["enabled"])
        self.assertEqual(audit["qa_issue_boundary_count_delta"], -1)
        self.assertEqual(len(audit["encode_settings_sha256"]), 64)
        self.assertEqual(audit["verified_technical_integrity_issue_count"], 0)

        def regressing_review(render_plan, cut_frame_index):
            payload = rendered_review(render_plan, cut_frame_index)
            if any(
                segment.boundary_look_track is not None
                for segment in render_plan.segments
            ):
                payload["discontinuities"]["color_shock"] = 0.32
            return payload

        rejected = verify_boundary_look_match_render_domain(
            plan,
            proposed,
            review_function=regressing_review,
        )
        self.assertEqual(rejected.applied_count, 0)
        self.assertEqual(rejected.render_rejected_count, 1)
        self.assertIn("hard_component_regression:color", rejected.decisions[0].reason)
        self.assertTrue(all(track is None for track in rejected.segment_tracks))

        for hazard in (
            "sharpness",
            "exposure",
            "mean_luma",
            "clipped_pixels",
            "structure_shock",
            "technical",
        ):
            with self.subTest(hazard=hazard):
                def hazardous_review(render_plan, cut_frame_index):
                    payload = rendered_review(render_plan, cut_frame_index)
                    adjusted = any(
                        segment.boundary_look_track is not None
                        for segment in render_plan.segments
                    )
                    if not adjusted:
                        return payload
                    if hazard == "sharpness":
                        payload["discontinuities"][
                            "sharpness_discontinuity"
                        ] = 0.14
                    elif hazard == "exposure":
                        payload["discontinuities"][
                            "exposure_discontinuity"
                        ] = 0.09
                    elif hazard == "mean_luma":
                        payload["frame_metrics"][2]["mean_luma"] = 0.46
                    elif hazard == "clipped_pixels":
                        payload["frame_metrics"][2][
                            "clipped_pixel_ratio"
                        ] = 0.020
                    elif hazard == "structure_shock":
                        payload["discontinuities"]["structure_shock"] = 0.32
                    else:
                        payload["technical_integrity_issue_reasons"] = [
                            "synthetic_flash_damage"
                        ]
                    return payload

                rejected_hazard = verify_boundary_look_match_render_domain(
                    plan,
                    proposed,
                    review_function=hazardous_review,
                )
                self.assertEqual(rejected_hazard.applied_count, 0)
                self.assertEqual(rejected_hazard.render_rejected_count, 1)
                verification = (
                    rejected_hazard.decisions[0].render_verification
                )
                self.assertIsNotNone(verification)
                assert verification is not None
                if hazard == "technical":
                    self.assertEqual(
                        verification.reason,
                        "technical_integrity_issue",
                    )
                else:
                    self.assertIn(hazard, verification.reason)

    def test_verified_ramp_keeps_neighboring_review_window_at_identity(self) -> None:
        plan = RenderPlan(
            sources=(Path("left.mp4"), Path("right.mp4")),
            segments=(
                RenderSegment(0, 0.0, 1.0, 0.0),
                RenderSegment(1, 0.0, 1.0, 1.0),
                RenderSegment(1, 1.0, 1.0, 2.0),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            width=1920,
            height=1080,
            fps=30.0,
        )
        proposed = analyze_boundary_look_match(
            plan,
            group_16x9_safe=True,
            measurement_cache=_MetricCache(  # type: ignore[arg-type]
                {
                    "left.mp4": tuple(
                        _metric(0.38, 0.30, 0.31) for _ in range(7)
                    ),
                    "right.mp4": tuple(
                        _metric(0.46, 0.35, 0.40) for _ in range(7)
                    ),
                }
            ),
        )

        def improving_review(render_plan, _cut_frame_index):
            adjusted = any(
                segment.boundary_look_track is not None
                for segment in render_plan.segments
            )
            return {
                "discontinuities": {
                    "sharpness_discontinuity": 0.10,
                    "exposure_discontinuity": 0.05 if adjusted else 0.08,
                    "color_shock": 0.24 if adjusted else 0.30,
                },
                "frame_metrics": [
                    {
                        "offset": offset,
                        "mean_luma": 0.40 if offset < 0 else 0.44,
                        "clipped_pixel_ratio": 0.002,
                    }
                    for offset in (-2, -1, 0, 1)
                ],
                "technical_integrity_issue_reasons": [],
                "issue_reasons": [],
            }

        admitted = verify_boundary_look_match_render_domain(
            plan,
            proposed,
            review_function=improving_review,
        )
        adjusted = admitted.apply(plan)
        middle_track = adjusted.segments[1].boundary_look_track
        self.assertIsNotNone(middle_track)
        assert middle_track is not None
        # Boundary 0 owns the first two frames of segment 1.  At the unrelated
        # boundary 1, its last two outgoing frames are exactly identity.
        self.assertTrue(middle_track.evaluate(28.0 / 30.0).is_identity)
        self.assertTrue(middle_track.evaluate(29.0 / 30.0).is_identity)
        self.assertIsNone(adjusted.segments[2].boundary_look_track)


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required")
class BoundaryLookRenderedRegressionTests(unittest.TestCase):
    def _source(self, path: Path, brightness: float, saturation: float) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=s=320x180:r=30:d=2",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000:duration=2",
                "-vf",
                f"eq=brightness={brightness}:saturation={saturation}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(path),
            ],
            check=True,
        )

    def test_actual_render_regression_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dark = root / "dark.mp4"
            bright = root / "bright.mp4"
            self._source(dark, -0.045, 0.90)
            self._source(bright, 0.045, 1.10)
            base = RenderPlan(
                sources=(dark, bright),
                segments=(
                    RenderSegment(0, 0.4, 0.7, 0.0),
                    RenderSegment(1, 0.4, 0.7, 0.7),
                ),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=root / "base.mp4",
                master_audio_start=0.0,
                width=854,
                height=480,
                fps=30.0,
            )
            match = analyze_boundary_look_match(
                base,
                group_16x9_safe=True,
                config=BoundaryLookMatchConfig(
                    maximum_luma_mad=0.06,
                    maximum_contrast_mad=0.10,
                    maximum_saturation_mad=0.12,
                    minimum_proxy_distance=0.04,
                ),
            )
            self.assertEqual(match.applied_count, 1)
            admitted = verify_boundary_look_match_render_domain(
                base,
                match,
            )
            self.assertEqual(admitted.applied_count, 0)
            verification = admitted.decisions[0].render_verification
            self.assertIsNotNone(verification)
            assert verification is not None
            self.assertFalse(verification.passed)
            self.assertGreater(
                verification.color_after,
                verification.color_before,
            )
            self.assertIn("hard_component_regression", verification.reason)
            self.assertEqual(
                verification.technical_integrity_issue_reasons,
                (),
            )


if __name__ == "__main__":
    unittest.main()
