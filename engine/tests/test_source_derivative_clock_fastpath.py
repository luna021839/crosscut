from fractions import Fraction
from pathlib import Path
import unittest
from unittest.mock import patch

from crosscut.seek_locality import ConstantFrameClock
from crosscut.source_derivative_diversity import (
    _nearest_pts_index, _probe_derivative_video_pts,
)


class SourceDerivativeClockFastpathTests(unittest.TestCase):
    def test_complete_verified_grid_avoids_decoding_for_timestamps(self):
        for rate in (Fraction(30000, 1001), Fraction(25), Fraction(60000, 1001)):
            with self.subTest(rate=rate), patch(
                'crosscut.source_derivative_diversity.probe_constant_frame_clock',
                return_value=ConstantFrameClock(rate, 300),
            ) as packet, patch(
                'crosscut.source_derivative_diversity._probe_actual_video_pts',
                side_effect=AssertionError('verified clock must not decode full video'),
            ):
                points, basis = _probe_derivative_video_pts(Path('/not-read'), timeout_seconds=120.)
                self.assertEqual(basis, 'verified_complete_cfr_packet_presentation_grid')
                self.assertEqual(points, tuple(float(f'{float(i/rate):.6f}') for i in range(300)))
                packet.assert_called_once_with(Path('/not-read'), timeout_seconds=10.)

    def test_unverified_grid_retains_original_decoded_pts_including_nonzero_origin(self):
        original = (1.4, 1.434, 1.483, 1.501)
        with patch('crosscut.source_derivative_diversity.probe_constant_frame_clock',
                   return_value=None), patch(
            'crosscut.source_derivative_diversity._probe_actual_video_pts', return_value=original,
        ) as decoded:
            points, basis = _probe_derivative_video_pts(Path('/not-read'), timeout_seconds=6.)
            self.assertIs(points, original)
            self.assertEqual(basis, 'ffprobe_best_effort_timestamp_time')
            decoded.assert_called_once_with(Path('/not-read'), timeout_seconds=6.)

    def test_failed_frame_probe_cannot_fabricate_a_grid(self):
        with patch('crosscut.source_derivative_diversity.probe_constant_frame_clock',
                   return_value=None), patch(
            'crosscut.source_derivative_diversity._probe_actual_video_pts',
            side_effect=RuntimeError('corrupt source'),
        ), self.assertRaisesRegex(RuntimeError, 'corrupt source'):
            _probe_derivative_video_pts(Path('/not-read'), timeout_seconds=5.)

    def test_exact_frame_selection_is_identical_around_half_frame_boundaries(self):
        rate = Fraction(30000, 1001)
        with patch('crosscut.source_derivative_diversity.probe_constant_frame_clock',
                   return_value=ConstantFrameClock(rate, 90)):
            points, _ = _probe_derivative_video_pts(Path('/not-read'), timeout_seconds=120.)
        decoded = tuple(float(f'{float(i/rate):.6f}') for i in range(90))
        for left, right in zip(decoded, decoded[1:]):
            for delta in (-1e-8, 0., 1e-8):
                time = (left+right)/2 + delta
                self.assertEqual(_nearest_pts_index(points, time), _nearest_pts_index(decoded, time))


if __name__ == '__main__':
    unittest.main()
