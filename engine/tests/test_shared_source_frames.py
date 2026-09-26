from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crosscut.native_focal_faces import NativeFocalFaceProvider
from crosscut.shared_source_frames import SharedSourceFrames, validate_shared_sources
from crosscut.source_first_focal_geometry import SourceFirstFocalGeometryProvider
from crosscut.visual_score import _VideoSampler
from crosscut.seek_locality import ConstantFrameClock
from engine.tests.test_seek_locality import FakeCapture


class Capture(FakeCapture):
    def __init__(self, _path=''):
        super().__init__()
        self.release_count = 0

    def isOpened(self):
        return not self.release_count

    def release(self):
        self.release_count += 1


class SharedSourceFrameTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.source = Path(directory.name)/'source.mp4'
        self.source.write_bytes(b'fixed source')
        self.captures = []
        def capture(path):
            value = Capture(path); self.captures.append(value); return value
        self.enterContext(patch('cv2.VideoCapture', side_effect=capture))
        self.enterContext(patch('crosscut.shared_source_frames.probe_constant_frame_clock',
            return_value=ConstantFrameClock(Fraction(30), 300)))

    def owner(self, **kwargs):
        owner = SharedSourceFrames(self.source, **kwargs)
        self.addCleanup(owner.close)
        return owner

    def test_native_and_two_working_consumers_decode_exact_pixels_once(self):
        owner = self.owner()
        native = NativeFocalFaceProvider((self.source,), shared_source_frames=(owner,))
        left = _VideoSampler(self.source, shared_source_frames=owner)
        right = _VideoSampler(self.source, shared_source_frames=owner)
        self.addCleanup(native.close); self.addCleanup(left.close); self.addCleanup(right.close)
        for index in (30, 31, 33, 32, 45, 46):
            image = native.frame_at(0, index/30)
            before = owner.reader.read_count
            np.testing.assert_array_equal(left.frame_at(index/30), image)
            np.testing.assert_array_equal(right.frame_at(index/30), image)
            self.assertEqual(owner.reader.read_count, before)
            self.assertEqual(image[0,0,0], index)
        self.assertEqual(sum(c.read_calls for c in self.captures[:-1]), 0)
        self.assertTrue(owner.reader.enabled)
        self.assertEqual(owner.reader.fallback_count, 0)

    def test_borrower_close_cannot_close_session_owned_capture(self):
        owner = self.owner()
        sampler = _VideoSampler(self.source, shared_source_frames=owner)
        native = NativeFocalFaceProvider((self.source,), shared_source_frames=(owner,))
        sampler.frame_at(1.)
        actual = owner._capture
        sampler.close(); native.close()
        self.assertEqual(actual.release_count, 0)
        np.testing.assert_array_equal(owner.native_at(2.), 60)
        owner.close(); owner.close()
        self.assertEqual(actual.release_count, 1)
        with self.assertRaises(RuntimeError):
            owner.working_at(1.)

    def test_small_working_cache_is_bounded_and_does_not_retain_native_bank(self):
        owner = self.owner(maximum_working_frames=16)
        for index in range(100):
            owner.native_at(index/30)
        self.assertEqual(len(owner._working), 16)
        self.assertEqual(len(owner.reader.frames), 1)
        self.assertFalse(owner.diagnostics()['raw_pixels_persisted'])

    def test_vfr_or_unverified_clock_never_aliases_nearby_times(self):
        owner = self.owner()
        with patch('crosscut.shared_source_frames.probe_constant_frame_clock', return_value=None):
            first = owner.native_at(1.00001)
            owner.native_at(1.00002)
        self.assertEqual(owner._capture.seek_calls, [1.00001*1000., 1.00002*1000.])
        self.assertIs(owner.working_at(1.00001), first)
        self.assertFalse(owner.reader.enabled)

    def test_changed_missing_or_retargeted_source_cannot_return_cached_pixels(self):
        owner = self.owner(); owner.native_at(1.)
        self.source.write_bytes(b'replaced source')
        with self.assertRaises(OSError):
            owner.working_at(1.)
        owner = self.owner(); owner.native_at(1.)
        self.source.unlink()
        with self.assertRaises(OSError):
            owner.working_at(1.)
        target = self.source.with_name('target'); target.write_bytes(b'target')
        self.source.symlink_to(target)
        owner = self.owner(); owner.native_at(1.)
        self.source.unlink(); self.source.write_bytes(b'another target')
        with self.assertRaises(OSError):
            owner.working_at(1.)

    def test_wrong_source_and_bad_queries_fail_before_pixel_reuse(self):
        owner = self.owner()
        other = self.source.with_name('other'); other.write_bytes(b'fixed source')
        with self.assertRaises(ValueError):
            _VideoSampler(other, shared_source_frames=owner)
        with self.assertRaises(ValueError):
            validate_shared_sources((self.source,), ())
        for seconds in (True, -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                owner.native_at(seconds)
        self.assertEqual(self.captures, [])

    def test_borrower_cache_hits_still_attest_source_storage(self):
        owner = self.owner()
        sampler = _VideoSampler(self.source, shared_source_frames=owner)
        native = NativeFocalFaceProvider((self.source,), shared_source_frames=(owner,))
        self.addCleanup(sampler.close); self.addCleanup(native.close)
        sampler.frame_at(1.)
        # Populate the native candidate LRU without invoking a face model.
        native._cache[(0, 1.)] = ()
        self.source.unlink()
        for cached_call in (lambda: sampler.frame_at(1.),
                sampler.assert_required_persistent_cache_available,
                lambda: native.candidates_at(0, 1.)):
            with self.assertRaises(OSError):
                cached_call()

    def test_real_focal_constructor_owns_shared_frames_and_cleans_failed_setup(self):
        with patch('crosscut.source_first_focal_geometry.BodyAppearanceEncoder'), \
             patch('crosscut.source_first_focal_geometry.BodyPose2DAnalyzer'):
            focal = SourceFirstFocalGeometryProvider((self.source,),
                body_model_path=Path('unused'), pose_model_path=Path('unused'))
            owner = focal.shared_source_frames[0]
            self.assertIs(focal.faces._samplers[0]._shared_source_frames, owner)
            self.assertIs(focal.native_faces._shared_source_frames[0], owner)
            focal.close()
            self.assertTrue(owner.closed)
        closed = []
        original = SharedSourceFrames.close
        def close(owner):
            closed.append(owner); original(owner)
        with patch.object(SharedSourceFrames, 'close', close), \
             patch('crosscut.source_first_focal_geometry.BodyPose2DAnalyzer', side_effect=RuntimeError('setup')):
            with self.assertRaisesRegex(RuntimeError, 'setup'):
                SourceFirstFocalGeometryProvider((self.source,),
                    body_model_path=Path('unused'), pose_model_path=Path('unused'))
        self.assertEqual(len(closed), 1)
        self.assertTrue(closed[0].closed)


if __name__ == '__main__':
    unittest.main()
