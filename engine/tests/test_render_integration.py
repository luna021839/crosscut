from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from crosscut.face_match import FaceMatchConfig, build_face_match_plan
from crosscut.ive import AspectRatio
from crosscut.render import (
    BoundaryLookKeyframe,
    BoundaryLookTrack,
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    NormalizedCrop,
    PanCropKeyframe,
    PanCropTrack,
    RenderPlan,
    RenderSegment,
    RenderTransition,
    TransitionKind,
    _boundary_look_filters,
    minimum_roll_safe_zoom,
    render,
)
from crosscut.timeline import ShotType
from crosscut.visual_score import VisualTimelineChoice, VisualTimelinePlan


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class RenderIntegrationTests(unittest.TestCase):
    def test_boundary_temperature_identity_tail_is_sample_exact(self) -> None:
        track = BoundaryLookTrack(
            (
                BoundaryLookKeyframe(0.0, temperature_balance=0.02),
                BoundaryLookKeyframe(1.0 / 30.0, temperature_balance=0.02),
                BoundaryLookKeyframe(7.0 / 30.0),
                BoundaryLookKeyframe(29.0 / 30.0),
            )
        )
        source = "testsrc2=size=160x90:rate=30:duration=1"

        def decode(filters: list[str]) -> np.ndarray:
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                source,
            ]
            if filters:
                command.extend(("-vf", ",".join(filters)))
            command.extend(("-pix_fmt", "yuv420p", "-f", "rawvideo", "-"))
            decoded = subprocess.run(
                command,
                check=True,
                capture_output=True,
            ).stdout
            return np.frombuffer(decoded, dtype=np.uint8).reshape(30, -1)

        baseline = decode([])
        adjusted = decode(_boundary_look_filters(track))

        self.assertGreater(np.count_nonzero(baseline[0] != adjusted[0]), 0)
        np.testing.assert_array_equal(adjusted[8:], baseline[8:])

    def _source(self, path: Path, color: str, duration: float = 1.0) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=160x90:r=30:d={duration}",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:sample_rate=48000:duration={duration}",
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

    def _boxed_source(self, path: Path) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=160x90:r=30:d=1,"
                "drawbox=x=64:y=28:w=24:h=20:color=white:t=fill",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000:duration=1",
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

    def _member_marker_source(self, path: Path, duration: float = 2.0) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c=magenta:s=160x90:r=30:d={duration}",
                "-f",
                "lavfi",
                "-i",
                f"color=c=white:s=24x28:r=30:d={duration}",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:sample_rate=48000:duration={duration}",
                "-filter_complex",
                "[0:v][1:v]overlay=x='20+50*t':y=28:shortest=1[v]",
                "-map",
                "[v]",
                "-map",
                "2:a",
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

    def test_many_fractional_frame_cuts_remain_one_second(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first, second = root / "first.mp4", root / "second.mp4"
            self._source(first, "red")
            self._source(second, "blue")
            output = root / "output.mp4"
            plan = RenderPlan(
                sources=(first, second),
                segments=tuple(
                    RenderSegment(
                        index % 2,
                        index / 12.0,
                        1.0 / 12.0,
                        index / 12.0,
                    )
                    for index in range(12)
                ),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=output,
                width=160,
                height=90,
                fps=30.0,
            )
            render(plan, video_codec="libx264")
            completed = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames,duration",
                    "-of",
                    "json",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            stream = json.loads(completed.stdout)["streams"][0]

        self.assertEqual(int(stream["nb_read_frames"]), 30)
        self.assertAlmostEqual(float(stream["duration"]), 1.0, places=5)

    def test_cut_boundaries_do_not_drift_between_colored_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first, second = root / "red.mp4", root / "blue.mp4"
            self._source(first, "red", duration=3.0)
            self._source(second, "blue", duration=3.0)
            output = root / "cuts.mp4"
            durations = (0.5, 0.7, 0.4, 0.6)
            master_start = 0.0
            segments: list[RenderSegment] = []
            for index, duration in enumerate(durations):
                segments.append(
                    RenderSegment(index % 2, master_start, duration, master_start)
                )
                master_start += duration
            plan = RenderPlan(
                sources=(first, second),
                segments=tuple(segments),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=output,
                width=160,
                height=90,
                fps=30.0,
            )

            render(plan, video_codec="libx264")
            decoded = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(output),
                    "-vf",
                    "scale=1:1,format=rgb24",
                    "-f",
                    "rawvideo",
                    "-",
                ],
                check=True,
                capture_output=True,
            ).stdout
            colors = [
                "red" if decoded[index] > decoded[index + 2] else "blue"
                for index in range(0, len(decoded), 3)
            ]
            transitions = [
                index
                for index in range(1, len(colors))
                if colors[index] != colors[index - 1]
            ]

        self.assertEqual(len(colors), 66)
        self.assertEqual(transitions, [15, 36, 48])

    def test_split_identity_same_source_render_matches_unsplit_moving_h264(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "moving-source.mp4"
            self._member_marker_source(source, duration=2.0)
            baseline_output = root / "unsplit.mp4"
            split_output = root / "split.mp4"
            durations = (0.2, 0.3, 0.4, 0.2, 0.3)
            source_start = 0.1
            master_start = 0.0
            split_segments: list[RenderSegment] = []
            for duration in durations:
                split_segments.append(
                    RenderSegment(
                        0,
                        source_start,
                        duration,
                        master_start,
                    )
                )
                source_start += duration
                master_start += duration
            transitions = tuple(
                RenderTransition(
                    tier=("D" if index % 2 == 0 else "C"),
                    reason=(
                        "same_source_hold_no_visible_switch"
                        if index % 2 == 0
                        else "same_source_internal_camera_cut"
                    ),
                )
                for index in range(len(split_segments) - 1)
            )
            baseline = RenderPlan(
                sources=(source,),
                segments=(RenderSegment(0, 0.1, 1.4, 0.0),),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=baseline_output,
                width=160,
                height=90,
                fps=30.0,
            )
            split = RenderPlan(
                sources=(source,),
                segments=tuple(split_segments),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=split_output,
                width=160,
                height=90,
                fps=30.0,
                transitions=transitions,
            )

            render(baseline, video_codec="libx264")
            render(split, video_codec="libx264")

            def decode(path: Path) -> np.ndarray:
                raw = subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(path),
                        "-pix_fmt",
                        "rgb24",
                        "-f",
                        "rawvideo",
                        "-",
                    ],
                    check=True,
                    capture_output=True,
                ).stdout
                return np.frombuffer(raw, dtype=np.uint8).reshape(
                    (-1, 90, 160, 3)
                )

            baseline_frames = decode(baseline_output)
            split_frames = decode(split_output)

        self.assertTrue(np.array_equal(split_frames, baseline_frames))
        self.assertEqual(len(split_frames), 42)
        boundary_frames = np.cumsum(
            [round(duration * 30.0) for duration in durations]
        )[:-1]
        for boundary_frame in boundary_frames:
            self.assertFalse(
                np.array_equal(
                    split_frames[boundary_frame - 1],
                    split_frames[boundary_frame],
                ),
                f"duplicated frame at split boundary {boundary_frame}",
            )

    def test_mixed_face_dissolve_chain_keeps_exact_frames_and_reference_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            red, blue, green = (
                root / "red.mp4",
                root / "blue.mp4",
                root / "green.mp4",
            )
            self._source(red, "red", duration=3.0)
            self._source(blue, "blue", duration=3.0)
            self._source(green, "green", duration=3.0)
            output = root / "mixed-transitions.mp4"
            plan = RenderPlan(
                sources=(red, blue, green),
                segments=(
                    RenderSegment(0, 0.0, 0.6, 0.0),
                    RenderSegment(1, 0.6, 0.7, 0.6),
                    RenderSegment(2, 1.3, 0.5, 1.3),
                ),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=output,
                width=160,
                height=90,
                fps=30.0,
                transitions=(
                    RenderTransition(
                        TransitionKind.FACE_DISSOLVE,
                        11,
                        "safe_face_evidence",
                        0.1,
                        "A",
                    ),
                    RenderTransition(tier="C", reason="hard_boundary"),
                ),
            )

            render(plan, video_codec="libx264")
            video_probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames,duration",
                    "-of",
                    "json",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            stream = json.loads(video_probe.stdout)["streams"][0]
            decoded_video = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(output),
                    "-vf",
                    "scale=1:1,format=rgb24",
                    "-f",
                    "rawvideo",
                    "-",
                ],
                check=True,
                capture_output=True,
            ).stdout
            pixels = np.frombuffer(decoded_video, dtype=np.uint8).reshape((-1, 3))
            decoded_audio = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(output),
                    "-map",
                    "0:a:0",
                    "-ac",
                    "1",
                    "-ar",
                    "48000",
                    "-f",
                    "f32le",
                    "-",
                ],
                check=True,
                capture_output=True,
            ).stdout
            audio = np.frombuffer(decoded_audio, dtype="<f4")

        self.assertEqual(int(stream["nb_read_frames"]), 54)
        self.assertAlmostEqual(float(stream["duration"]), 1.8, places=5)
        # The nominal boundary starts a gradual red-to-blue blend, and the
        # later hard edge switches directly to green.
        self.assertGreater(int(pixels[17, 0]), int(pixels[17, 2]) * 3)
        self.assertGreater(int(pixels[23, 0]), 25)
        self.assertGreater(int(pixels[23, 2]), 25)
        self.assertGreater(int(pixels[29, 2]), int(pixels[29, 0]) * 3)
        self.assertGreater(int(pixels[38, 2]), int(pixels[38, 1]) * 2)
        self.assertGreater(int(pixels[39, 1]), int(pixels[39, 2]))
        # Audio is one uninterrupted reference-source trim, not per-shot audio.
        for boundary_seconds in (0.6, 1.3):
            center = round(boundary_seconds * 48_000)
            window = audio[center - 1_200 : center + 1_200]
            self.assertGreater(float(np.sqrt(np.mean(window * window))), 0.02)

    def test_dynamic_member_pan_filter_renders_real_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            self._source(source, "purple")
            output = root / "portrait.mp4"
            track = PanCropTrack(
                (
                    PanCropKeyframe(
                        0.0, NormalizedCrop(0.0, 0.0, 0.31640625, 1.0)
                    ),
                    PanCropKeyframe(
                        0.5, NormalizedCrop(0.34, 0.0, 0.31640625, 1.0)
                    ),
                    PanCropKeyframe(
                        1.0, NormalizedCrop(0.68359375, 0.0, 0.31640625, 1.0)
                    ),
                )
            )
            plan = RenderPlan(
                sources=(source,),
                segments=(
                    RenderSegment(0, 0.0, 1.0, 0.0, pan_crop=track),
                ),
                reference_source=0,
                aspect_ratio=AspectRatio.PORTRAIT,
                output=output,
                width=90,
                height=160,
                fps=30.0,
            )

            render(plan, video_codec="libx264")
            completed = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames,width,height",
                    "-of",
                    "json",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            stream = json.loads(completed.stdout)["streams"][0]

        self.assertEqual(int(stream["nb_read_frames"]), 30)
        self.assertEqual((stream["width"], stream["height"]), (90, 160))

    def test_face_match_pan_crop_renders_aligned_enlarged_subject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "boxed.mp4"
            self._boxed_source(source)
            output = root / "face-match.mp4"
            face_box = (64 / 160, 28 / 90, 24 / 160, 20 / 90)
            choice = VisualTimelineChoice(
                segment_index=0,
                source_index=0,
                canonical_master_start=0.0,
                canonical_master_end=1.0,
                source_local_start=0.0,
                source_local_end=1.0,
                quality_score=0.8,
                composition_score=0.8,
                boundary_score=0.8,
                shot_scale=0.4,
                shot_type=ShotType.MEDIUM,
                shot_id="0:shot-0",
                face_samples_xywh=(face_box, face_box, face_box),
            )
            match = build_face_match_plan(
                visual_timeline=VisualTimelinePlan((choice,), 1.0),
                source_aspect_ratios=(16 / 9,),
                config=FaceMatchConfig(target_face_height_override=0.28),
            )
            self.assertEqual(match.applied_count, 1)
            plan = RenderPlan(
                sources=(source,),
                segments=(
                    RenderSegment(
                        0,
                        0.0,
                        1.0,
                        0.0,
                        pan_crop=match.pan_crop_for(0),
                    ),
                ),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=output,
                width=160,
                height=90,
                fps=30.0,
            )
            render(plan, video_codec="libx264")
            decoded = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(output),
                    "-frames:v",
                    "1",
                    "-pix_fmt",
                    "gray",
                    "-f",
                    "rawvideo",
                    "-",
                ],
                check=True,
                capture_output=True,
            ).stdout
            frame = np.frombuffer(decoded, dtype=np.uint8).reshape((90, 160))
            ys, xs = np.where(frame > 180)

        self.assertGreaterEqual(int(ys.max() - ys.min() + 1), 24)
        self.assertAlmostEqual(float(xs.mean() / 160), 0.5, delta=0.035)
        self.assertAlmostEqual(float(ys.mean() / 90), 0.42, delta=0.035)

    def test_dense_120_keyframe_member_pan_is_simplified_and_renders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            self._source(source, "purple", duration=4.0)
            output = root / "dense-pan.mp4"
            track = PanCropTrack(
                tuple(
                    PanCropKeyframe(
                        index / 30.0,
                        NormalizedCrop(
                            0.0 if index % 2 == 0 else 0.68359375,
                            0.0,
                            0.31640625,
                            1.0,
                        ),
                    )
                    for index in range(120)
                )
            )
            plan = RenderPlan(
                sources=(source,),
                segments=(RenderSegment(0, 0.0, 4.0, 0.0, pan_crop=track),),
                reference_source=0,
                aspect_ratio=AspectRatio.PORTRAIT,
                output=output,
                width=90,
                height=160,
                fps=30.0,
            )

            render(plan, video_codec="libx264")
            completed = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "default=nokey=1:noprint_wrappers=1",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertEqual(int(completed.stdout.strip()), 120)

    def test_group_transform_preserves_timewarp_frames_and_hides_roll_corners(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            self._source(source, "white", duration=2.02)
            output = root / "group-transform.mp4"
            safe_zoom = minimum_roll_safe_zoom(1.0)
            track = EditorialTransformTrack(
                (
                    EditorialTransformKeyframe(0.0, safe_zoom, 0.50, 0.50, -1.0),
                    EditorialTransformKeyframe(1.0, 1.10, 0.50, 0.50, 0.0),
                    EditorialTransformKeyframe(2.0, 1.16, 0.50, 0.50, 1.0),
                ),
                reason="integration_motion_proof",
            )
            plan = RenderPlan(
                sources=(source,),
                segments=(
                    RenderSegment(
                        0,
                        0.0,
                        2.0,
                        0.0,
                        editorial_transform=track,
                        source_duration=2.02,
                    ),
                ),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=output,
                width=160,
                height=90,
                fps=30.0,
            )

            render(plan, video_codec="libx264")
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames,duration",
                    "-of",
                    "json",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            decoded = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(output),
                    "-pix_fmt",
                    "rgb24",
                    "-f",
                    "rawvideo",
                    "-",
                ],
                check=True,
                capture_output=True,
            ).stdout
            frames = np.frombuffer(decoded, dtype=np.uint8).reshape((-1, 90, 160, 3))
            border = np.concatenate(
                (
                    frames[:, :2].reshape(-1, 3),
                    frames[:, -2:].reshape(-1, 3),
                    frames[:, :, :2].reshape(-1, 3),
                    frames[:, :, -2:].reshape(-1, 3),
                ),
                axis=0,
            )

        self.assertEqual(int(stream["nb_read_frames"]), 60)
        self.assertAlmostEqual(float(stream["duration"]), 2.0, places=5)
        self.assertEqual(float(np.mean(np.max(border, axis=1) < 8)), 0.0)

    def test_member_pan_crop_composes_with_portrait_editorial_camera(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "member.mp4"
            # Consume 2.02 source seconds into 2.00 output seconds so this one
            # integration covers the real local-timewarp + moving tracker path.
            self._member_marker_source(source, duration=2.02)
            output = root / "member-editorial.mp4"
            crop_width = (9.0 / 16.0) / (16.0 / 9.0)
            pan = PanCropTrack(
                (
                    PanCropKeyframe(
                        0.0,
                        NormalizedCrop(0.04, 0.0, crop_width, 1.0),
                    ),
                    PanCropKeyframe(
                        2.0,
                        NormalizedCrop(0.66, 0.0, crop_width, 1.0),
                    ),
                )
            )
            editorial = EditorialTransformTrack(
                (
                    EditorialTransformKeyframe(0.0, 1.18, 0.52, 0.5, -1.6),
                    EditorialTransformKeyframe(0.75, 1.0, 0.5, 0.5, 0.0),
                    EditorialTransformKeyframe(2.0, 1.08, 0.5, 0.5, 0.0),
                ),
                reason="confirmed_member_track_plus_editorial_motion",
            )
            plan = RenderPlan(
                sources=(source,),
                segments=(
                    RenderSegment(
                        0,
                        0.0,
                        2.0,
                        0.0,
                        pan_crop=pan,
                        editorial_transform=editorial,
                        source_duration=2.02,
                    ),
                ),
                reference_source=0,
                aspect_ratio=AspectRatio.PORTRAIT,
                output=output,
                width=90,
                height=160,
                fps=30.0,
            )

            render(plan, video_codec="libx264")
            decoded = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(output),
                    "-pix_fmt",
                    "rgb24",
                    "-f",
                    "rawvideo",
                    "-",
                ],
                check=True,
                capture_output=True,
            ).stdout
            frames = np.frombuffer(decoded, dtype=np.uint8).reshape((-1, 160, 90, 3))
            bright_masks = np.min(frames, axis=3) > 220
            bright_counts = np.sum(bright_masks, axis=(1, 2))
            bright_margins: list[tuple[int, int, int, int]] = []
            for mask in bright_masks:
                rows, columns = np.where(mask)
                self.assertGreater(len(rows), 0)
                bright_margins.append(
                    (
                        int(columns.min()),
                        89 - int(columns.max()),
                        int(rows.min()),
                        159 - int(rows.max()),
                    )
                )
            border = np.concatenate(
                (
                    frames[:, :2].reshape(-1, 3),
                    frames[:, -2:].reshape(-1, 3),
                    frames[:, :, :2].reshape(-1, 3),
                    frames[:, :, -2:].reshape(-1, 3),
                ),
                axis=0,
            )

        self.assertEqual(len(frames), 60)
        self.assertGreater(int(np.min(bright_counts)), 1500)
        self.assertGreaterEqual(min(min(item) for item in bright_margins), 2)
        self.assertEqual(float(np.mean(np.max(border, axis=1) < 8)), 0.0)


if __name__ == "__main__":
    unittest.main()
