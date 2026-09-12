from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

from crosscut.seek_locality import (
    ConstantFrameClock,
    SeekLocalityReader,
    constant_frame_clock_from_probe,
    probe_constant_frame_clock,
)


def payload(count=12):
    return {
        'streams': [{
            'codec_name': 'av1', 'r_frame_rate': '30000/1001',
            'avg_frame_rate': '30000/1001', 'time_base': '1/30000',
            'start_time': '0.000000', 'nb_frames': str(count),
        }],
        'packets': [{'pts': n * 1001, 'duration': 1001} for n in range(count)],
    }


class FakeCapture:
    def __init__(self, *, fps=30.0, count=300, backend='FFMPEG'):
        self.fps = fps
        self.count = count
        self.backend = backend
        self.next_index = 0
        self.last_pts = -1.0
        self.seek_calls = []
        self.read_calls = 0
        self.bad_pts = False
        self.fail_next_read = False

    def getBackendName(self):
        return self.backend

    def get(self, key):
        return {
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_COUNT: self.count,
            cv2.CAP_PROP_POS_FRAMES: float(self.next_index),
            cv2.CAP_PROP_POS_MSEC: self.last_pts + (50.0 if self.bad_pts else 0),
        }[key]

    def set(self, key, value):
        assert key == cv2.CAP_PROP_POS_MSEC
        self.seek_calls.append(value)
        self.next_index = min(self.count, int(value / 1000.0 * self.fps + .5))
        return True

    def read(self):
        self.read_calls += 1
        if self.fail_next_read:
            self.fail_next_read = False
            return False, None
        if self.next_index >= self.count:
            return False, None
        index = self.next_index
        self.next_index += 1
        self.last_pts = index / self.fps * 1000.0
        return True, np.full((2, 3, 3), index, dtype=np.int32)


class ConstantFrameClockTests(unittest.TestCase):
    def test_complete_grid_and_reordered_packets_are_accepted(self):
        original = payload()
        self.assertEqual(
            constant_frame_clock_from_probe(original),
            ConstantFrameClock(Fraction(30000, 1001), 12),
        )
        original['packets'].reverse()
        self.assertIsNotNone(constant_frame_clock_from_probe(original))

    def test_average_rate_does_not_authorize_vfr(self):
        value = payload()
        value['packets'][4]['pts'] += 1
        self.assertIsNone(constant_frame_clock_from_probe(value))

    def test_missing_duplicate_gap_and_incomplete_durations_are_rejected(self):
        cases = []
        value = payload(); value['packets'].pop(); cases.append(value)
        value = payload(); value['packets'][3] = value['packets'][2]; cases.append(value)
        value = payload(); value['packets'][3]['pts'] += 1001; cases.append(value)
        value = payload(); del value['packets'][2]['duration']; cases.append(value)
        value = payload(); value['packets'][2]['duration'] = 1000; cases.append(value)
        value = payload(); value['packets'][2]['pts'] = True; cases.append(value)
        for value in cases:
            with self.subTest(value=value):
                self.assertIsNone(constant_frame_clock_from_probe(value))

    def test_bad_stream_metadata_is_not_certified(self):
        for field, value in (
            ('codec_name', 'h264'), ('start_time', '1.0'),
            ('avg_frame_rate', '0/0'), ('r_frame_rate', '30/1'),
            ('time_base', '0/1'), ('nb_frames', 'N/A'), ('nb_frames', 12.5),
        ):
            sample = payload()
            sample['streams'][0][field] = value
            with self.subTest(field=field):
                self.assertIsNone(constant_frame_clock_from_probe(sample))

    def test_missing_source_or_probe_fails_to_optional_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(probe_constant_frame_clock(Path(directory) / 'absent'))


class SeekLocalityTests(unittest.TestCase):
    def make_reader(self, capture=None, **kwargs):
        capture = capture or FakeCapture()
        return SeekLocalityReader(
            capture, clock=ConstantFrameClock(Fraction(30, 1), 300),
            process_frame=lambda frame: frame, **kwargs,
        )

    def test_adjacent_and_repeated_pixel_indices_do_not_seek_again(self):
        reader = self.make_reader()
        for index in (20, 21, 22, 20, 24, 21):
            np.testing.assert_array_equal(reader.frame_at(index / 30), index)
        self.assertEqual(reader.seek_count, 1)
        self.assertEqual(reader.read_count, 5)
        self.assertEqual(reader.pixel_cache_hit_count, 2)

    def test_fractional_requests_match_double_timestamp_seek(self):
        reader = self.make_reader()
        baseline = FakeCapture()
        for value in (0, 0.5, 1.5, 2.5, 19.5, 20.5, 20.5 + 1e-12,
                      20.5 - 1e-12, 20.1, 20.2):
            seconds = value / 30
            baseline.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            _, expected = baseline.read()
            np.testing.assert_array_equal(reader.frame_at(seconds), expected)

    def test_large_jumps_backward_jumps_and_eviction_keep_pixel_identity(self):
        reader = self.make_reader(maximum_cached_frames=3, maximum_forward_frames=4)
        for index in (20, 23, 24, 20, 150, 151, 149, 298):
            np.testing.assert_array_equal(reader.frame_at(index / 30), index)
            self.assertLessEqual(len(reader.frames), 3)
        self.assertEqual(reader.seek_count, 5)

    def test_irregular_clock_or_backend_uses_timestamp_path(self):
        for capture, clock in (
            (FakeCapture(), None),
            (FakeCapture(backend='AVFOUNDATION'), ConstantFrameClock(Fraction(30), 300)),
            (FakeCapture(fps=29.97), ConstantFrameClock(Fraction(30), 300)),
            (FakeCapture(count=299), ConstantFrameClock(Fraction(30), 300)),
        ):
            reader = SeekLocalityReader(capture, clock=clock, process_frame=lambda f: f)
            self.assertFalse(reader.enabled)
            reader.frame_at(.1)
            reader.frame_at(.11)
            self.assertEqual(reader.seek_count, 2)

    def test_observed_timestamp_divergence_disables_reuse(self):
        capture = FakeCapture()
        reader = self.make_reader(capture)
        reader.frame_at(1)
        capture.bad_pts = True
        np.testing.assert_array_equal(reader.frame_at(31 / 30), 31)
        self.assertFalse(reader.enabled)
        self.assertEqual(reader.fallback_count, 1)
        self.assertFalse(reader.frames)

    def test_external_decoder_motion_disables_reuse_on_cache_miss(self):
        capture = FakeCapture()
        reader = self.make_reader(capture)
        reader.frame_at(1)
        capture.read()
        np.testing.assert_array_equal(reader.frame_at(32 / 30), 32)
        self.assertFalse(reader.enabled)

    def test_eof_and_negative_times_follow_legacy_behavior(self):
        reader = self.make_reader()
        np.testing.assert_array_equal(reader.frame_at(-1), 0)
        self.assertIsNone(reader.frame_at(100))
        np.testing.assert_array_equal(reader.frame_at(1), 30)

    def test_failure_never_returns_a_partially_decoded_neighbor(self):
        capture = FakeCapture()
        reader = self.make_reader(capture)
        reader.frame_at(1)
        capture.fail_next_read = True
        np.testing.assert_array_equal(reader.frame_at(31 / 30), 31)
        self.assertFalse(reader.enabled)

    def test_non_finite_timestamps_rejected_without_decoder_work(self):
        reader = self.make_reader()
        for seconds in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(ValueError):
                reader.frame_at(seconds)
        self.assertEqual(reader.seek_count, 0)


class ActualDecoderLocalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ffmpeg = shutil.which('ffmpeg')
        cls.ffprobe = shutil.which('ffprobe')
        if not cls.ffmpeg or not cls.ffprobe:
            raise unittest.SkipTest('ffmpeg and ffprobe required')
        encoders = subprocess.run(
            [cls.ffmpeg, '-hide_banner', '-encoders'], capture_output=True,
            check=True, timeout=10,
        ).stdout
        if b'libsvtav1' not in encoders:
            raise unittest.SkipTest('SVT-AV1 fixture encoder unavailable')
        directory = tempfile.TemporaryDirectory(prefix='crosscut-seek-clock-')
        cls.addClassCleanup(directory.cleanup)
        cls.directory = Path(directory.name)
        cls.files = {}
        for name, rate, filters in (
            ('cfr', '30', []),
            ('ntsc', '30000/1001', []),
            ('vfr', '30', ['-vf', r'select=not(eq(mod(n\,5)\,0))',
                           '-fps_mode', 'vfr']),
        ):
            output = cls.directory / (name + '.mp4')
            subprocess.run(
                [cls.ffmpeg, '-v', 'error', '-f', 'lavfi', '-i',
                 'testsrc2=size=96x128:rate=' + rate + ':duration=2',
                 *filters, '-an', '-c:v', 'libsvtav1', '-preset', '13',
                 '-crf', '40', '-g', '30', '-svtav1-params', 'lp=1',
                 str(output)], capture_output=True, check=True, timeout=30,
            )
            cls.files[name] = output

    def compare_requests(self, path, clock):
        baseline = cv2.VideoCapture(str(path))
        accelerated = cv2.VideoCapture(str(path))
        self.addCleanup(baseline.release)
        self.addCleanup(accelerated.release)
        if accelerated.getBackendName() != 'FFMPEG':
            self.skipTest('FFmpeg decoder backend required')
        reader = SeekLocalityReader(
            accelerated, clock=clock, process_frame=lambda frame: frame,
        )
        rate = baseline.get(cv2.CAP_PROP_FPS)
        values = [0, -1, 1, 2, 2.49, 2.5, 2.51, 3, 8, 12, 5, 30, 32, 31,
                  48, 49, 55, 51, 56, 20, 21, 100, 1]
        for value in values:
            seconds = value / rate
            baseline.set(cv2.CAP_PROP_POS_MSEC, max(0, seconds) * 1000)
            success, expected = baseline.read()
            actual = reader.frame_at(seconds)
            with self.subTest(value=value):
                if success:
                    np.testing.assert_array_equal(actual, expected)
                else:
                    self.assertIsNone(actual)
        return reader

    def test_real_constant_rate_av1_matches_timestamp_decoder(self):
        for name in ('cfr', 'ntsc'):
            with self.subTest(name=name):
                clock = probe_constant_frame_clock(
                    self.files[name], ffprobe=self.ffprobe,
                )
                self.assertIsNotNone(clock)
                reader = self.compare_requests(self.files[name], clock)
                self.assertLess(reader.seek_count, 23)
                self.assertGreater(reader.pixel_cache_hit_count, 0)

    def test_real_vfr_is_not_aliasable_and_preserves_original_seeking(self):
        clock = probe_constant_frame_clock(self.files['vfr'], ffprobe=self.ffprobe)
        self.assertIsNone(clock)
        reader = self.compare_requests(self.files['vfr'], clock)
        self.assertFalse(reader.enabled)
        self.assertEqual(reader.seek_count, 23)


if __name__ == '__main__':
    unittest.main()
