from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from crosscut.post_render_perceptual_quality import (
    PerceptualQualityConfig,
    _parse_plan,
    _retention_candidates,
    evaluate_post_render_perceptual_quality,
)


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray] | None, fps: float) -> None:
        self.frames = frames
        self.fps = fps
        self.position = 0
        self.released = False

    def isOpened(self) -> bool:
        return self.frames is not None

    def get(self, property_id: int) -> float:
        if self.frames is None or not self.frames:
            return 0.0
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.frames[0].shape[1])
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.frames[0].shape[0])
        if property_id == cv2.CAP_PROP_FPS:
            return self.fps
        if property_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self.position)
        return 0.0

    def set(self, property_id: int, value: float) -> bool:
        if property_id != cv2.CAP_PROP_POS_FRAMES or self.frames is None:
            return False
        self.position = max(0, int(round(value)))
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.frames is None or self.position >= len(self.frames):
            return (False, None)
        frame = self.frames[self.position]
        self.position += 1
        return (True, frame.copy())

    def release(self) -> None:
        self.released = True


def _texture(*, brightness: int = 0) -> np.ndarray:
    y, x = np.indices((54, 96))
    checker = (((x // 3) + (y // 3)) % 2) * 100
    ramp = (x * 70 // 95)
    frame = np.empty((54, 96, 3), dtype=np.uint8)
    frame[..., 0] = np.clip(25 + checker + ramp + brightness, 0, 255)
    frame[..., 1] = np.clip(45 + checker + ramp + brightness, 0, 255)
    frame[..., 2] = np.clip(65 + checker + ramp + brightness, 0, 255)
    return frame


def _plan(root: Path, *, single_source: bool = False) -> dict[str, object]:
    fps = 10.0
    sources = [str(root / "source-0.mp4")]
    source_indices = [0]
    source_starts = [0.0]
    if not single_source:
        sources.append(str(root / "source-1.mp4"))
        source_indices = [0, 1, 0]
        source_starts = [0.0, 0.0, 2.0]
    segments = [
        {
            "source_index": source_index,
            "source_start": source_start,
            "source_local_start": source_start,
            "source_local_end": source_start + 2.0,
            "source_playback_rate": 1.0,
            "duration": 2.0,
            "master_start": index * 2.0,
            "crop": None,
            "pan_crop": None,
            "editorial_transform": None,
            "look_adjustment": None,
            "boundary_look_track": None,
        }
        for index, (source_index, source_start) in enumerate(
            zip(source_indices, source_starts, strict=True)
        )
    ]
    return {
        "sources": sources,
        "segments": segments,
        "transitions": [
            {"kind": "hard_cut", "frame_count": 0}
            for _ in range(len(segments) - 1)
        ],
        "fps": fps,
        "width": 96,
        "height": 54,
        "output_frame_count": len(segments) * 20,
    }


def _settings() -> PerceptualQualityConfig:
    return PerceptualQualityConfig(
        analysis_width=96,
        minimum_same_run_comparable_pair_count=4,
        minimum_per_source_output_frame_count=4,
        retention_samples_per_second=2.0,
        maximum_retention_samples_per_source=8,
        minimum_retention_samples_per_source=2,
        retention_segment_guard_frames=2,
    )


class PostRenderPerceptualQualityTests(unittest.TestCase):
    def _evaluate(
        self,
        *,
        output_frames: list[np.ndarray],
        source_zero: list[np.ndarray],
        source_one: list[np.ndarray] | None,
        single_source: bool = False,
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output.mp4"
            plan = _plan(root, single_source=single_source)
            frames_by_path = {
                str(output): output_frames,
                str(root / "source-0.mp4"): source_zero,
                str(root / "source-1.mp4"): source_one,
            }

            def factory(path: str) -> _FakeCapture:
                return _FakeCapture(frames_by_path.get(path), 10.0)

            return evaluate_post_render_perceptual_quality(
                output,
                plan,
                config=_settings(),
                capture_factory=factory,
            )

    def test_stable_exact_source_render_passes_with_scalar_only_evidence(self) -> None:
        frame = _texture()
        source_zero = [frame.copy() for _ in range(40)]
        source_one = [frame.copy() for _ in range(20)]
        output = source_zero[:20] + source_one + source_zero[20:]

        payload = self._evaluate(
            output_frames=output,
            source_zero=source_zero,
            source_one=source_one,
        )

        self.assertEqual(payload["verdict"], "PASS")
        self.assertTrue(payload["passed"])
        metrics = payload["metrics"]
        self.assertEqual(metrics["video"]["decoded_frame_count"], 60)
        self.assertEqual(
            metrics["exact_source_retention"]["adequately_covered_source_count"],
            2,
        )
        self.assertAlmostEqual(
            metrics["exact_source_retention"]["detail_retention_ratio"]["median"],
            1.0,
        )
        self.assertFalse(
            payload["serialization_policy"]["raw_frames_serialized"]
        )
        self.assertFalse(
            payload["serialization_policy"]["per_frame_descriptors_serialized"]
        )

    def test_blurred_final_render_fails_exact_source_retention(self) -> None:
        frame = _texture()
        source_zero = [frame.copy() for _ in range(40)]
        source_one = [frame.copy() for _ in range(20)]
        clean = source_zero[:20] + source_one + source_zero[20:]
        blurred = [cv2.GaussianBlur(item, (11, 11), 3.0) for item in clean]

        payload = self._evaluate(
            output_frames=blurred,
            source_zero=source_zero,
            source_one=source_one,
        )

        self.assertEqual(payload["verdict"], "FAIL")
        self.assertIn(
            "perceptual_detail_retention_p10",
            {item["code"] for item in payload["issues"]},
        )

    def test_same_source_exposure_flicker_is_separate_fail_axis(self) -> None:
        source_frame = _texture()
        source = [source_frame.copy() for _ in range(20)]
        output = [
            _texture(brightness=45 if index % 2 else -25)
            for index in range(20)
        ]

        payload = self._evaluate(
            output_frames=output,
            source_zero=source,
            source_one=None,
            single_source=True,
        )

        self.assertEqual(payload["verdict"], "FAIL")
        self.assertIn(
            "perceptual_luma_high_step_pumping",
            {item["code"] for item in payload["issues"]},
        )
        self.assertGreater(
            payload["metrics"]["temporal_pumping"]["high_step_fraction"]["luma"],
            0.5,
        )

    def test_missing_used_source_is_never_reported_as_pass(self) -> None:
        frame = _texture()
        source_zero = [frame.copy() for _ in range(40)]
        output = [frame.copy() for _ in range(60)]

        payload = self._evaluate(
            output_frames=output,
            source_zero=source_zero,
            source_one=None,
        )

        self.assertEqual(payload["verdict"], "FAIL")
        self.assertIn(
            "perceptual_source_video_unavailable_1",
            {item["code"] for item in payload["issues"]},
        )

    def test_dissolve_overlap_is_never_mapped_as_single_incoming_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan = _plan(Path(directory))
        plan["transitions"][0] = {
            "kind": "face_dissolve",
            "frame_count": 6,
        }
        fps, _width, _height, _count, _sources, spans = _parse_plan(plan)
        config = PerceptualQualityConfig(
            analysis_width=96,
            retention_samples_per_second=10.0,
            maximum_retention_samples_per_source=100,
            retention_segment_guard_frames=2,
        )

        candidates = _retention_candidates(spans, fps=fps, config=config)

        incoming = spans[1]
        self.assertEqual(incoming.incoming_transition_frames, 6)
        self.assertGreaterEqual(
            min(
                index
                for indices in candidates.values()
                for index in indices
                if incoming.start_frame <= index < incoming.end_frame
            ),
            incoming.start_frame + 6 + 2,
        )

    def test_dynamic_crop_face_pan_and_look_tracks_keep_retention_coverage(self) -> None:
        frame = _texture()
        source_zero = [frame.copy() for _ in range(40)]
        source_one = [frame.copy() for _ in range(20)]
        output = [frame.copy() for _ in range(60)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "output.mp4"
            plan = _plan(root)
            segments = plan["segments"]
            pan_crop = {
                "keyframes": [
                    {
                        "time_seconds": 0.0,
                        "crop": {"x": 0.05, "y": 0.05, "width": 0.90, "height": 0.90},
                    },
                    {
                        "time_seconds": 2.0,
                        "crop": {"x": 0.07, "y": 0.05, "width": 0.90, "height": 0.90},
                    },
                ]
            }
            for segment in segments[:2]:
                segment["pan_crop"] = pan_crop
                segment["look_adjustment"] = {
                    "brightness": 0.02,
                    "contrast": 1.04,
                    "saturation": 0.96,
                    "temperature_balance": 0.01,
                }
                segment["boundary_look_track"] = {
                    "keyframes": [
                        {
                            "time_seconds": 0.0,
                            "brightness": 0.01,
                            "contrast": 1.02,
                            "saturation": 0.98,
                        },
                        {
                            "time_seconds": 2.0,
                            "brightness": 0.0,
                            "contrast": 1.0,
                            "saturation": 1.0,
                        },
                    ]
                }
            segments[2]["editorial_transform"] = {
                "reason": "stage_edit_dynamic_crop_head_v2_stage_edit_preserved",
                "keyframes": [
                    {
                        "time_seconds": 0.0,
                        "zoom": 1.08,
                        "center_x": 0.51,
                        "center_y": 0.49,
                        "roll_degrees": 0.4,
                    },
                    {
                        "time_seconds": 2.0,
                        "zoom": 1.12,
                        "center_x": 0.52,
                        "center_y": 0.50,
                        "roll_degrees": 0.0,
                    },
                ],
            }
            segments[2]["look_adjustment"] = {
                "brightness": -0.02,
                "contrast": 0.96,
                "saturation": 1.04,
                "temperature_balance": 0.0,
            }
            plan["transitions"][0] = {
                "kind": "face_dissolve",
                "frame_count": 4,
            }
            frames_by_path = {
                str(output_path): output,
                str(root / "source-0.mp4"): source_zero,
                str(root / "source-1.mp4"): source_one,
            }

            def factory(path: str) -> _FakeCapture:
                return _FakeCapture(frames_by_path.get(path), 10.0)

            payload = evaluate_post_render_perceptual_quality(
                output_path,
                plan,
                config=_settings(),
                capture_factory=factory,
            )

        retention = payload["metrics"]["exact_source_retention"]
        self.assertEqual(retention["adequately_covered_source_count"], 2)
        self.assertEqual(retention["geometry_reprojection_failure_count"], 0)
        self.assertGreater(retention["comparable_sample_count"], 0)
        self.assertNotIn(
            "perceptual_exact_source_retention_unavailable",
            {item["code"] for item in payload["issues"]},
        )


if __name__ == "__main__":
    unittest.main()
