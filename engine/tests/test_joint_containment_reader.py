"""Tracking raster inputs must not change when decoder locality is reused."""
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosscut.joint_boundary_provider import DecodedBoundaryOptionProvider
from crosscut.seek_locality import ConstantFrameClock, SeekLocalityReader
from crosscut.visual_boundary_repair import BoundaryDescriptorCache
from crosscut.visual_feature_cache import CanonicalFrameIdentity
from crosscut.visual_score import _working_frame
from engine.tests.test_seek_locality import FakeCapture


class NativeCapture(FakeCapture):
    def __init__(self):
        super().__init__()
        self.releases = 0
        self.native = np.random.default_rng(7101).integers(
            0, 256, size=(720, 1280, 3), dtype=np.uint8)

    def read(self):
        success, frame = super().read()
        return success, self.native if success else None

    def release(self):
        self.releases += 1


def make_provider(**kwargs):
    return DecodedBoundaryOptionProvider(
        sources=(Path('source-a.mp4'), Path('source-b.mp4')),
        audio_offsets=(0., 0.), scene_cut_times=((), ()),
        source_timewarps=None, descriptor_cache=BoundaryDescriptorCache(),
        output_fps=30., **kwargs)


def identity(seconds):
    return CanonicalFrameIdentity.from_seconds(seconds, frame_rate=30.)


class JointContainmentReaderTests(unittest.TestCase):
    def test_production_native_reader_reuses_adjacent_frames_per_source(self):
        captures = (NativeCapture(), NativeCapture())
        captures[1].native = 255 - captures[1].native
        with make_provider() as provider, \
                patch.object(provider, '_sampler', return_value=SimpleNamespace(frame_identity=identity)), \
                patch('crosscut.joint_boundary_provider.cv2.VideoCapture', side_effect=captures), \
                patch('crosscut.joint_boundary_provider.probe_constant_frame_clock', create=True,
                      return_value=ConstantFrameClock(Fraction(30), 300)) as probe, \
                patch('crosscut.joint_boundary_provider._detect_faces', return_value=((), 'test')):
            for frame_index in range(30, 43):
                for source_index in (0, 1):
                    frame, _, _ = provider._cut_containment_pixels(source_index, frame_index / 30.)
                    np.testing.assert_array_equal(frame, _working_frame(
                        captures[source_index].native, maximum_width=960))
            self.assertEqual([len(c.seek_calls) for c in captures], [1, 1])
            self.assertEqual(probe.call_count, 2)
            for reader in provider._containment_readers.values():
                self.assertEqual(reader.maximum_cached_frames, 1)
                self.assertLessEqual(len(reader.frames), 1)
        self.assertEqual([c.releases for c in captures], [1, 1])
        self.assertFalse(provider._containment_readers)

    def test_unverified_clock_keeps_original_timestamp_seeks(self):
        capture = NativeCapture()
        with make_provider() as provider, \
                patch.object(provider, '_sampler', return_value=SimpleNamespace(frame_identity=identity)), \
                patch('crosscut.joint_boundary_provider.cv2.VideoCapture', return_value=capture), \
                patch('crosscut.joint_boundary_provider.probe_constant_frame_clock', create=True,
                      return_value=None), \
                patch('crosscut.joint_boundary_provider._detect_faces', return_value=((), 'test')):
            requests = (1.123456, 1.156789, 1.190122)
            for seconds in requests:
                provider._cut_containment_pixels(0, seconds)
            self.assertEqual(capture.seek_calls,
                [identity(t).canonical_seconds * 1000. for t in requests])

    def test_native_flow_and_detection_are_independent_resizes(self):
        capture = NativeCapture()
        with make_provider() as provider, \
                patch.object(provider, '_sampler', return_value=SimpleNamespace(frame_identity=identity)), \
                patch('crosscut.joint_boundary_provider.cv2.VideoCapture', return_value=capture), \
                patch('crosscut.joint_boundary_provider._detect_faces', return_value=((), 'test')) as detect:
            frame, boxes, backend = provider._cut_containment_pixels(0, 1.12345)
            np.testing.assert_array_equal(frame, _working_frame(capture.native, maximum_width=960))
            observed = detect.call_args.args[0]
            np.testing.assert_array_equal(observed, _working_frame(capture.native))
            self.assertFalse(np.array_equal(observed, _working_frame(frame)))
            self.assertEqual((boxes, backend), ((), 'test'))
            self.assertIs(provider._cut_containment_pixels(0, 1.12345)[0], frame)
            self.assertEqual(detect.call_count, 1)
        self.assertEqual(capture.releases, 1)
        provider.close()
        self.assertEqual(capture.releases, 1)
        self.assertEqual(provider._containment_pixel_bytes, 0)

    def test_injected_raster_preserves_seconds_and_does_not_open_video(self):
        capture = NativeCapture()
        requests = []
        def supplied(source, seconds):
            requests.append((source, seconds))
            return capture.native
        with make_provider(descriptor_provider=lambda *_: None,
                           containment_frame_provider=supplied) as provider, \
                patch('crosscut.joint_boundary_provider.cv2.VideoCapture') as video, \
                patch('crosscut.joint_boundary_provider._detect_faces', return_value=((), 'test')):
            provider._cut_containment_pixels(1, 1.123456789)
            self.assertEqual(requests, [(1, 1.123456789)])
            video.assert_not_called()

    def test_missing_injected_frame_is_not_cached_as_success(self):
        with make_provider(descriptor_provider=lambda *_: None,
                           containment_frame_provider=lambda *_: None) as provider, \
                patch('crosscut.joint_boundary_provider._detect_faces') as detect:
            self.assertEqual(provider._cut_containment_pixels(0, 1.), (None, (), 'none'))
            self.assertFalse(provider._containment_pixels)
            detect.assert_not_called()

    def test_storage_guard_runs_even_before_pixel_cache_hit(self):
        capture = NativeCapture()
        with make_provider(descriptor_provider=lambda *_: None,
                           containment_frame_provider=lambda *_: capture.native) as provider, \
                patch('crosscut.joint_boundary_provider._detect_faces', return_value=((), 'test')):
            provider._cut_containment_pixels(0, 1.)
            with patch.object(provider, '_assert_required_persistent_cache_available',
                              side_effect=OSError('storage unavailable')):
                with self.assertRaisesRegex(OSError, 'storage unavailable'):
                    provider._cut_containment_pixels(0, 1.)

    def test_single_native_frame_cache_handles_reverse_and_repeated_requests(self):
        baseline, capture = FakeCapture(), FakeCapture()
        reader = SeekLocalityReader(capture,
            clock=ConstantFrameClock(Fraction(30), 300),
            process_frame=lambda frame: frame, maximum_cached_frames=1)
        for index in (21,22,23,22,21,21,22,40,41,90,91,90,89,299,300,0):
            seconds = identity(index / 30.).canonical_seconds
            baseline.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.)
            ok, expected = baseline.read()
            actual = reader.frame_at(seconds)
            if ok:
                np.testing.assert_array_equal(actual, expected)
            else:
                self.assertIsNone(actual)
            self.assertLessEqual(len(reader.frames), 1)
        self.assertLess(reader.seek_count, len(baseline.seek_calls))


if __name__ == '__main__':
    unittest.main()
