"""Actual encoder metamorphic check: slicing must not change framing or clocks."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

from crosscut.ive import AspectRatio
from crosscut.render import (RenderPlan, RenderSegment, RenderTransition,
    EditorialTransformTrack, EditorialTransformKeyframe, NormalizedCrop,
    PanCropTrack, PanCropKeyframe, SourceLookAdjustment, TransitionKind,
    MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY, DIRECT_MEMBER_POSE_DISSOLVE_REASON, build_filter_graph, render)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class RenderSplitGeometryTests(unittest.TestCase):
    def test_split_preserves_outgoing_dissolve_camera_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for sw, sh, width, height in ((321, 241, 180, 320), (240, 320, 320, 180)):
                source = root / f'source-{sw}.mp4'
                subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    f'testsrc=s={sw}x{sh}:r=30:d=3', '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=mono',
                    '-shortest', '-c:v', 'libx264', '-pix_fmt', 'yuv444p', '-c:a', 'aac', str(source)],
                    check=True, capture_output=True)
                def pixels(path):
                    return subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path),
                        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], check=True, capture_output=True).stdout
                for nominal in (EditorialTransformTrack((EditorialTransformKeyframe(0., 1.),)),
                                EditorialTransformTrack((EditorialTransformKeyframe(0., 1.1),))):
                    zoom = 1.1 if nominal is not None and nominal.keyframes[0].zoom > 1 else 1.
                    handle = EditorialTransformTrack((EditorialTransformKeyframe(0., zoom),
                        EditorialTransformKeyframe(6/30, 1.25, .53, .48, 1.5)))
                    segment = RenderSegment(0, 0., 1., 0., editorial_transform=nominal,
                        outgoing_overlap_transform=handle)
                    incoming = RenderSegment(1, 1., 1., 1.)
                    base = RenderPlan((source, source), (segment, incoming), 0,
                        AspectRatio.PORTRAIT if height > width else AspectRatio.LANDSCAPE,
                        root / 'base.mp4', width=width, height=height, fps=30.,
                        transitions=(RenderTransition(TransitionKind.FACE_DISSOLVE, 7,
                            DIRECT_MEMBER_POSE_DISSOLVE_REASON, tier='A',
                            transform_overlap_authority=MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY),))
                    render(base, video_codec='libx264')
                    expected = pixels(base.output)
                    self.assertEqual(len(expected), 60*width*height*3)
                    for pieces in (2, 3):
                        with self.subTest(size=(sw, sh), nominal=nominal, pieces=pieces):
                            split = replace(base, output=root / 'split.mp4',
                                segments=tuple(replace(segment, source_start=i/pieces, master_start=i/pieces,
                                    duration=1/pieces, outgoing_overlap_transform=handle if i == pieces-1 else None)
                                    for i in range(pieces)) + (incoming,),
                                transitions=(RenderTransition(),)*(pieces-1)+base.transitions)
                            render(split, video_codec='libx264')
                            self.assertEqual(pixels(split.output), expected)

    def test_split_preserves_effective_geometry_and_clock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for source_size, output_size in (((640, 480), (320, 180)),
                                             ((3840, 2160), (360, 640)),
                                             ((321, 241), (320, 180))):
                sw, sh = source_size
                width, height = output_size
                source = root / f"{sw}-{sh}.mp4"
                # The offset marker distinguishes contain and aspect-fill.
                filters = f"color=c=0x404040:s={sw}x{sh}:r=30:d=0.8,format=yuv444p,drawbox=x=iw*.52:y=ih*.38:w=iw*.03:h=ih*.06:color=red:t=fill"
                subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", filters,
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-shortest",
                    "-c:v", "libx264", "-crf", "10", "-c:a", "aac", str(source)],
                    check=True, capture_output=True)
                source_metadata = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_streams", "-of", "json", str(source)], capture_output=True, check=True).stdout)["streams"][0]
                self.assertEqual((source_metadata["width"], source_metadata["height"]), source_size)
                for geometry in ("absent", "neutral", "zoom", "crop", "pan", "neutral_look"):
                    track = (EditorialTransformTrack((EditorialTransformKeyframe(
                        0., 1.2 if geometry == "zoom" else 1., .5, .5, 0.),))
                        if geometry in {"neutral", "zoom", "neutral_look"} else None)
                    crop = NormalizedCrop(.1, .1, .8, .8) if geometry == "crop" else None
                    pan = PanCropTrack((PanCropKeyframe(0., NormalizedCrop(.1, .1, .8, .8)),)) if geometry == "pan" else None
                    segment = RenderSegment(0, 0., .6, 0., crop=crop, pan_crop=pan,
                        editorial_transform=track,
                        look_adjustment=SourceLookAdjustment(brightness=.01) if geometry == "neutral_look" else None)
                    base = RenderPlan((source,), (segment,), 0,
                        AspectRatio.PORTRAIT if height > width else AspectRatio.LANDSCAPE,
                        root / f"base-{sw}-{geometry}.mp4", width=width, height=height, fps=30.)
                    render(base, video_codec="libx264")
                    def decode(path):
                        capture = cv2.VideoCapture(str(path))
                        frames = []
                        try:
                            while True:
                                ok, frame = capture.read()
                                if not ok: break
                                frames.append(frame)
                        finally:
                            capture.release()
                        return np.asarray(frames)
                    expected = decode(base.output)
                    self.assertEqual(len(expected), 18)
                    for pieces in (2, 3):
                        with self.subTest(source=source_size, geometry=geometry, pieces=pieces):
                            duration = .6 / pieces
                            plan = replace(base, output=root / f"split-{sw}-{geometry}-{pieces}.mp4",
                                segments=tuple(replace(segment, source_start=i*duration,
                                    master_start=i*duration, duration=duration) for i in range(pieces)),
                                transitions=tuple(RenderTransition() for _ in range(pieces-1)))
                            self.assertEqual(build_filter_graph(plan).count("[0:v]trim="), 1)
                            render(plan, video_codec="libx264")
                            actual = decode(plan.output)
                            np.testing.assert_array_equal(actual, expected)
                            metadata = json.loads(subprocess.run(["ffprobe", "-v", "error",
                                "-select_streams", "v:0", "-show_streams", "-show_frames", "-of", "json",
                                str(plan.output)], capture_output=True, check=True).stdout)
                            self.assertEqual(len(metadata["frames"]), 18)
                            self.assertEqual(metadata["streams"][0]["avg_frame_rate"], "30/1")
                            for index, frame in enumerate(metadata["frames"]):
                                self.assertAlmostEqual(float(frame["best_effort_timestamp_time"]), index/30., delta=.00005)
                neutral = EditorialTransformTrack((EditorialTransformKeyframe(0., 1., .5, .5, 0.),))
                mixed = replace(base, output=root / f"mixed-{sw}.mp4",
                    segments=tuple(RenderSegment(0, i*.2, .2, i*.2,
                        editorial_transform=neutral if i == 1 else None) for i in range(3)),
                    transitions=(RenderTransition(), RenderTransition()))
                self.assertEqual(build_filter_graph(mixed).count("[0:v]trim="), 1)
                render(mixed, video_codec="libx264")
                mixed_frames = decode(mixed.output)
                def centroid(frame):
                    ys, xs = np.nonzero((frame[:, :, 2] > 140) & (frame[:, :, 0] < 90) & (frame[:, :, 1] < 90))
                    self.assertGreater(len(xs), 5)
                    return np.array([xs.mean(), ys.mean()])
                absent, filled = decode(root / f"base-{sw}-absent.mp4"), decode(root / f"base-{sw}-neutral.mp4")
                for index, frame in enumerate(mixed_frames):
                    reference = filled[index] if 6 <= index < 12 else absent[index]
                    self.assertLess(np.linalg.norm(centroid(frame)-centroid(reference)), 2.)
