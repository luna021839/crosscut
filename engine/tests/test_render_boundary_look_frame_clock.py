"""Actual FFmpeg equivalence of coalesced and explicitly split color ramps."""
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from crosscut.ive import AspectRatio
from crosscut.render import (
    BoundaryLookKeyframe, BoundaryLookTrack, EditorialTransformKeyframe,
    EditorialTransformTrack, RenderPlan, RenderSegment, render,
)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class RenderBoundaryLookFrameClockTests(unittest.TestCase):
    def test_fractional_frame_boundary_look_survives_both_coalescing_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "gray.mp4"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "color=c=0x505050:s=160x90:r=30:d=1", "-f", "lavfi", "-i",
                "anullsrc=r=48000:cl=mono", "-shortest", "-c:v", "libx264",
                "-c:a", "aac", str(source),
            ], check=True, capture_output=True, timeout=15)
            ramp = BoundaryLookTrack((
                BoundaryLookKeyframe(0.),
                BoundaryLookKeyframe(1 / 30., brightness=.02),
                BoundaryLookKeyframe(.10, brightness=.04),
            ))
            for prefix in (False, True):
                with self.subTest(shared_clock_slices=prefix):
                    start = .1 if prefix else 0.
                    segments = (
                        RenderSegment(0, .2 + start, .05, start),
                        RenderSegment(0, .25 + start, .15, .05 + start,
                                      boundary_look_track=ramp),
                    )
                    if prefix:
                        segments = (RenderSegment(0, .2, .1, 0., editorial_transform=
                            EditorialTransformTrack((EditorialTransformKeyframe(0., 1.10),))),) + segments
                    optimized = RenderPlan(
                        (source,) * len(segments), segments, 0, AspectRatio.LANDSCAPE,
                        root / "optimized.mp4", master_audio_start=.2,
                        width=160, height=90, fps=30.,
                    )
                    # Identical static media with distinct input indices forces
                    # segment-local execution without private-function stubs.
                    reference = replace(optimized, output=root / "reference.mp4",
                        segments=tuple(replace(segment, source_index=index)
                                       for index, segment in enumerate(segments)))
                    frames = []
                    for plan in (reference, optimized):
                        render(plan, video_codec="libx264", maximum_runtime_seconds=15.)
                        pixels = subprocess.run([
                            "ffmpeg", "-v", "error", "-i", str(plan.output),
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
                        ], check=True, capture_output=True, timeout=15).stdout
                        decoded = np.frombuffer(pixels, dtype=np.uint8).reshape(-1, 90, 160, 3)
                        self.assertEqual(len(decoded), plan.output_frame_count)
                        frames.append(decoded)
                    np.testing.assert_array_equal(frames[1], frames[0], err_msg=
                        f"reference={frames[0][:, 40, 70, 0].tolist()} optimized={frames[1][:, 40, 70, 0].tolist()}")


if __name__ == "__main__":
    unittest.main()
