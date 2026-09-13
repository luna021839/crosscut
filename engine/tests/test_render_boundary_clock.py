from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import unittest
import shutil
import subprocess
import tempfile

import numpy as np

from crosscut.render import (
    EditorialTransformKeyframe, EditorialTransformTrack, RenderPlan, RenderSegment,
    RenderTransition, TransitionKind, render,
)
from crosscut.ive import AspectRatio
from crosscut.render_boundary_clock import hard_source_cut_native_clocks
from crosscut.render_frame_schedule import VerifiedRenderFrameClock, probe_render_frame_clock


def plan(segments, fps=29.97):
    return RenderPlan(sources=(Path('a.mp4'), Path('b.mp4'), Path('c.mp4')),
        segments=tuple(segments), reference_source=0, aspect_ratio=AspectRatio.LANDSCAPE,
        output=Path('unused.mp4'), fps=fps)


def clocks(fps=Fraction(30000, 1001), count=8000):
    return {i: VerifiedRenderFrameClock(fps, count, 'a' * 64) for i in range(3)}


class RenderBoundaryClockTests(unittest.TestCase):
    def test_real_diagnostic_frame_ids_not_half_frame_measurement_ids(self):
        durations = (1.2912101444991322, .9909098441988533, 1.3913102445992251)
        starts = (165.4449617696613, 146.1688541852568, 210.55250905535212)
        segments = tuple(RenderSegment(i, start, duration, sum(durations[:i]))
                         for i, (start, duration) in enumerate(zip(starts, durations)))
        result = hard_source_cut_native_clocks(plan(segments), clocks())
        self.assertEqual([cut.first_incoming_output_frame for cut in result], [39, 68])
        self.assertEqual([(cut.outgoing.native_frame_index, cut.incoming.native_frame_index)
                          for cut in result], [(4996, 4381), (4409, 6310)])
        half_frame_measurement = starts[0] + durations[0] - .5 / 29.97
        self.assertEqual(int(half_frame_measurement * 30000 / 1001 + .5), 4997)
        self.assertNotEqual(result[0].outgoing.native_frame_index, 4997)

    def test_contiguous_same_source_run_uses_one_native_clock(self):
        segments = (RenderSegment(0, .0143, .084, 0.),
                    RenderSegment(0, .0983, .084, .084),
                    RenderSegment(1, .021, 1., .168))
        result = hard_source_cut_native_clocks(plan(segments, 30.), clocks(Fraction(60)))
        self.assertEqual(len(result), 1)
        cut = result[0]
        self.assertEqual(cut.outgoing_segment_index, 1)
        self.assertEqual(cut.outgoing.unit_first_segment_index, 0)
        self.assertEqual(cut.first_incoming_output_frame, 5)
        self.assertEqual(cut.outgoing.native_frame_index, 9)
        self.assertEqual(cut.incoming.native_frame_index, 1)

    def test_shared_clock_with_different_camera_tracks_preserves_native_indices(self):
        transform = EditorialTransformTrack((EditorialTransformKeyframe(0., 1.2),
                                             EditorialTransformKeyframe(.084, 1.2)))
        segments = (RenderSegment(0, .0143, .084, 0., editorial_transform=transform),
                    RenderSegment(0, .0983, .084, .084), RenderSegment(1, .021, 1., .168))
        result = hard_source_cut_native_clocks(plan(segments, 30.), clocks(Fraction(60)))
        self.assertEqual(result[0].outgoing.native_frame_index, 9)
        self.assertEqual(result[0].outgoing.unit_first_segment_index, 0)

    def test_missing_clock_never_falls_back_to_an_approximate_frame(self):
        p = plan((RenderSegment(0, 0., 1., 0.), RenderSegment(1, 0., 1., 1.)))
        with self.assertRaisesRegex(ValueError, 'no verified'):
            hard_source_cut_native_clocks(p, {0: clocks()[0]})

    def test_terminal_clone_is_not_reported_as_new_observed_media(self):
        p = plan((RenderSegment(0, 0., 1.1, 0.), RenderSegment(1, 0., 1., 1.1)), 30.)
        result = hard_source_cut_native_clocks(p, clocks(Fraction(30), 30))
        self.assertTrue(result[0].outgoing.terminal_clone)
        self.assertEqual(result[0].outgoing.native_frame_index, 29)
        self.assertFalse(result[0].incoming.terminal_clone)

    def test_wrong_plan_type_is_rejected(self):
        with self.assertRaises(TypeError):
            hard_source_cut_native_clocks({}, clocks())

    def test_overlap_is_not_mislabeled_as_a_single_native_frame(self):
        p = plan((RenderSegment(0, 0., 1., 0.), RenderSegment(1, 0., 1., 1.)))
        p = replace(p, transitions=(RenderTransition(TransitionKind.FACE_DISSOLVE, 4),))
        with self.assertRaisesRegex(ValueError, 'overlap pixels'):
            hard_source_cut_native_clocks(p, clocks())


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class RenderBoundaryClockPixelTests(unittest.TestCase):
    def test_reported_neighbors_are_the_frames_emitted_by_the_real_renderer(self):
        frames = np.zeros((100, 90, 160, 3), dtype=np.uint8)
        for index in range(100):
            for bit in range(8):
                frames[index, :, bit*20:(bit+1)*20, :] = 235 if index >> bit & 1 else 16
        with tempfile.TemporaryDirectory(prefix='crosscut-native-boundary-test-') as directory:
            root = Path(directory)
            source = root / 'synthetic-frame-ids.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                '-video_size', '160x90', '-framerate', '60', '-i', 'pipe:0',
                '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=mono', '-c:v', 'libx264',
                '-crf', '0', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(source)],
                input=frames.tobytes(), capture_output=True, check=True)
            clock = probe_render_frame_clock(source)
            self.assertIsNotNone(clock)
            for camera in (False, True):
                track = EditorialTransformTrack((EditorialTransformKeyframe(0., 1.04),
                    EditorialTransformKeyframe(.084, 1.04))) if camera else None
                segments = (RenderSegment(0, .0143, .084, 0., editorial_transform=track),
                    RenderSegment(0, .0983, .084, .084), RenderSegment(1, .35, .4, .168))
                p = replace(plan(segments, 30.), sources=(source, source), width=160, height=90,
                            output=root / f'cut-{camera}.mp4')
                expected = hard_source_cut_native_clocks(p, {0: clock, 1: clock})[0]
                render(p, video_codec='libx264')
                raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(p.output), '-an',
                    '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], capture_output=True, check=True).stdout
                actual = np.frombuffer(raw, np.uint8).reshape((-1, 90, 160, 3))
                self.assertEqual(len(actual), p.output_frame_count)
                for sample in (expected.outgoing, expected.incoming):
                    frame = actual[sample.output_frame_index]
                    observed = sum((frame[40:50, bit*20+8:bit*20+12].mean() > 125) << bit
                                   for bit in range(8))
                    self.assertEqual(observed, sample.native_frame_index, (camera, sample))


if __name__ == '__main__':
    unittest.main()
