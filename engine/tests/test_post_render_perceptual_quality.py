from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

from crosscut.ive import AspectRatio
from crosscut.post_render_perceptual_quality import (
    PerceptualQualityConfig,
    _parse_plan,
    _rendered_source_frame_index,
    _retention_candidates,
    evaluate_post_render_perceptual_quality,
)
from crosscut.render import RenderPlan, RenderSegment, render


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

    def test_same_detail_but_wrong_final_geometry_fails_structure_binding(
        self,
    ) -> None:
        frame = _texture()
        source_zero = [frame.copy() for _ in range(40)]
        source_one = [frame.copy() for _ in range(20)]
        clean = source_zero[:20] + source_one + source_zero[20:]
        shifted = [np.roll(item, 20, axis=1) for item in clean]

        payload = self._evaluate(
            output_frames=shifted,
            source_zero=source_zero,
            source_one=source_one,
        )

        self.assertEqual(payload["verdict"], "FAIL")
        self.assertIn(
            "perceptual_exact_structure_similarity_p10",
            {item["code"] for item in payload["issues"]},
        )
        self.assertGreater(
            payload["metrics"]["exact_source_retention"]
            ["detail_retention_ratio"]["p10"],
            0.95,
        )

    def test_one_previously_capped_segment_geometry_tamper_now_fails(
        self,
    ) -> None:
        fps = 30.0
        segment_frame_count = 28
        segment_duration = segment_frame_count / fps
        segment_count = 12
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.mp4"
            output_path = root / "output.mp4"
            segments = [
                {
                    "source_index": 0,
                    "source_start": index * segment_duration,
                    "source_local_start": index * segment_duration,
                    "source_local_end": (index + 1) * segment_duration,
                    "source_playback_rate": 1.0,
                    "duration": segment_duration,
                    "master_start": index * segment_duration,
                    "crop": None,
                    "pan_crop": None,
                    "editorial_transform": None,
                    "look_adjustment": None,
                    "boundary_look_track": None,
                }
                for index in range(segment_count)
            ]
            plan = {
                "sources": [str(source_path)],
                "segments": segments,
                "transitions": [
                    {"kind": "hard_cut", "frame_count": 0}
                    for _ in range(segment_count - 1)
                ],
                "fps": fps,
                "width": 96,
                "height": 54,
                "output_frame_count": segment_count * segment_frame_count,
            }
            config = PerceptualQualityConfig(
                analysis_width=96,
                minimum_same_run_comparable_pair_count=4,
                minimum_per_source_output_frame_count=4,
                retention_samples_per_second=1.0,
                maximum_retention_samples_per_source=4,
                minimum_retention_samples_per_source=2,
                retention_segment_guard_frames=3,
            )
            parsed_fps, _width, _height, _count, _sources, spans = (
                _parse_plan(plan)
            )

            # Reproduce the former source-wide cap to prove the selected target
            # had no structure sample before mandatory per-segment allocation.
            stride = int(round(parsed_fps / config.retention_samples_per_second))
            old_candidates: list[int] = []
            for span in spans:
                first = span.start_frame + config.retention_segment_guard_frames
                last = span.end_frame - config.retention_segment_guard_frames - 1
                old_candidates.append((first + last) // 2)
                aligned = ((first + stride - 1) // stride) * stride
                old_candidates.extend(range(aligned, last + 1, stride))
            old_candidates = sorted(set(old_candidates))
            old_limited = [
                old_candidates[
                    round(
                        index
                        * (len(old_candidates) - 1)
                        / (config.maximum_retention_samples_per_source - 1)
                    )
                ]
                for index in range(
                    config.maximum_retention_samples_per_source
                )
            ]
            old_covered = {
                next(
                    span.segment_index
                    for span in spans
                    if span.start_frame <= frame_index < span.end_frame
                )
                for frame_index in old_limited
            }
            target = next(
                span for span in spans if span.segment_index not in old_covered
            )
            self.assertAlmostEqual(
                (target.end_frame - target.start_frame) / fps,
                0.9333333333333333,
            )

            new_candidates = _retention_candidates(
                spans,
                fps=parsed_fps,
                config=config,
            )
            new_covered = {
                next(
                    span.segment_index
                    for span in spans
                    if span.start_frame <= frame_index < span.end_frame
                )
                for frame_indices in new_candidates.values()
                for frame_index in frame_indices
            }
            self.assertEqual(new_covered, set(range(segment_count)))
            self.assertLessEqual(
                len(new_candidates[0]),
                segment_count
                + config.maximum_retention_samples_per_source,
            )

            frame = _texture()
            source_frames = [
                frame.copy()
                for _ in range(segment_count * segment_frame_count + 4)
            ]
            output_frames = [item.copy() for item in source_frames[: -4]]
            output_frames[target.start_frame : target.end_frame] = [
                np.roll(item, 20, axis=1)
                for item in output_frames[
                    target.start_frame : target.end_frame
                ]
            ]
            frames_by_path = {
                str(source_path): source_frames,
                str(output_path): output_frames,
            }

            def factory(path: str) -> _FakeCapture:
                return _FakeCapture(frames_by_path.get(path), fps)

            payload = evaluate_post_render_perceptual_quality(
                output_path,
                plan,
                config=config,
                capture_factory=factory,
            )

        retention = payload["metrics"]["exact_source_retention"]
        self.assertEqual(retention["render_segment_count"], segment_count)
        self.assertEqual(
            retention["structure_covered_segment_count"],
            segment_count,
        )
        target_evidence = retention["per_segment"][target.segment_index]
        self.assertGreaterEqual(target_evidence["requested_sample_count"], 1)
        self.assertGreaterEqual(
            target_evidence["structure_comparable_sample_count"],
            1,
        )
        self.assertEqual(payload["verdict"], "FAIL")
        self.assertIn(
            "perceptual_exact_structure_segment_binding_failed",
            {item["code"] for item in payload["issues"]},
        )

    def test_segment_without_safe_interior_structure_sample_fails_closed(
        self,
    ) -> None:
        frame = _texture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "output.mp4"
            source_path = root / "source.mp4"
            plan = {
                "sources": [str(source_path)],
                "segments": [
                    {
                        "source_index": 0,
                        "source_start": 0.0,
                        "source_local_start": 0.0,
                        "source_local_end": 0.4,
                        "source_playback_rate": 1.0,
                        "duration": 0.4,
                        "master_start": 0.0,
                        "crop": None,
                        "pan_crop": None,
                        "editorial_transform": None,
                        "look_adjustment": None,
                        "boundary_look_track": None,
                    }
                ],
                "transitions": [],
                "fps": 10.0,
                "width": 96,
                "height": 54,
                "output_frame_count": 4,
            }
            frames_by_path = {
                str(output_path): [frame.copy() for _ in range(4)],
                str(source_path): [frame.copy() for _ in range(4)],
            }

            def factory(path: str) -> _FakeCapture:
                return _FakeCapture(frames_by_path.get(path), 10.0)

            payload = evaluate_post_render_perceptual_quality(
                output_path,
                plan,
                config=PerceptualQualityConfig(
                    analysis_width=96,
                    minimum_same_run_comparable_pair_count=2,
                    minimum_per_source_output_frame_count=4,
                    maximum_retention_samples_per_source=4,
                    minimum_retention_samples_per_source=1,
                    retention_segment_guard_frames=3,
                ),
                capture_factory=factory,
            )

        evidence = payload["metrics"]["exact_source_retention"]
        self.assertFalse(evidence["structure_segment_coverage_complete"])
        self.assertEqual(evidence["uncovered_segment_indices"], [0])
        self.assertEqual(
            evidence["per_segment"][0]["incomparability_reason"],
            "no_non_overlap_interior_frame_after_transition_and_guard",
        )
        self.assertIn(
            "perceptual_exact_structure_segment_coverage_incomplete",
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

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required")
    def test_real_ffmpeg_non_aligned_trim_passes_but_one_frame_shift_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "alternating-geometry-source.mp4"
            normal = root / "normal.mp4"
            shifted = root / "one-frame-shifted.mp4"
            # Each geometry persists for two frames.  That leaves enough stable
            # same-run pairs for pumping QA while making a systematic one-frame
            # substitution unambiguously different at every retention sample.
            video_filter = (
                "nullsrc=size=320x180:rate=30:duration=4,"
                "geq=lum='if(eq(mod(floor((N+1)/2),2),0),"
                "if(between(X,16,104)*between(Y,16,164),235,16),"
                "if(between(X,216,304)*between(Y,16,164),235,16))':"
                "cb=128:cr=128"
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    video_filter,
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=48000:duration=4",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "12",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(source),
                ],
                check=True,
            )
            render_plan = RenderPlan(
                sources=(source,),
                segments=(RenderSegment(0, 0.01, 3.0, 0.0),),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=normal,
                width=320,
                height=180,
                fps=30.0,
            )
            render(render_plan, video_codec="libx264")
            plan = json.loads(json.dumps(render_plan.to_json()))
            span = _parse_plan(plan)[5][0]

            # Legacy timestamp trim starts at frame 1. The current verified
            # sampler instead starts at frame 0; the render sidecar tells QA
            # which clock was actually used. Shift the fixture's two-frame
            # geometry phase so its sampled even ticks straddle a change.
            self.assertEqual(
                _rendered_source_frame_index(
                    span,
                    relative_time=1.0,
                    source_fps=30.0,
                ),
                31,
            )
            self.assertEqual(round((0.01 + 1.0) * 30.0), 30)
            normal_payload = evaluate_post_render_perceptual_quality(
                normal,
                plan,
            )
            self.assertEqual(normal_payload["verdict"], "PASS")

            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(normal),
                    "-vf",
                    (
                        "trim=start_frame=1,setpts=PTS-STARTPTS,"
                        "tpad=stop_mode=clone:stop_duration=1,"
                        "trim=end_frame=90,setpts=N/(30*TB)"
                    ),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "12",
                    "-c:a",
                    "copy",
                    str(shifted),
                ],
                check=True,
            )
            shifted_payload = evaluate_post_render_perceptual_quality(
                shifted,
                plan,
            )

        self.assertEqual(shifted_payload["verdict"], "FAIL")
        issue_codes = {item["code"] for item in shifted_payload["issues"]}
        self.assertIn(
            "perceptual_exact_structure_similarity_p10",
            issue_codes,
        )
        self.assertIn(
            "perceptual_exact_structure_similarity_median",
            issue_codes,
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
