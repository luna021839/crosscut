from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from crosscut.reference_frame_decode_cache import SequentialReferenceFrameCache
from crosscut.reference_match import ReferenceMatchConfig,decode_target_frames_sequential


class ReferenceFrameDecodeCacheTests(unittest.TestCase):
    def fixture(self,root,name='source.avi'):
        path=Path(root)/name
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),30.,(96,64))
        if not writer.isOpened():
            self.skipTest('MJPG fixture encoder unavailable')
        try:
            for i in range(45):
                rng=np.random.default_rng(i)
                writer.write(rng.integers(0,255,(64,96,3),dtype=np.uint8))
        finally:
            writer.release()
        return path

    def test_forward_overlap_reverse_and_eof_match_fresh_sequential(self):
        with tempfile.TemporaryDirectory() as root, SequentialReferenceFrameCache() as cache:
            path=self.fixture(root)
            for indices in ((2,8),(8,12),(12,20),(1,20),(40,44,50)):
                expected=decode_target_frames_sequential(path,indices)
                actual=cache.decode(path,indices)
                self.assertEqual(actual.requested_indices,expected.requested_indices)
                self.assertEqual(actual.frame_rate,expected.frame_rate)
                self.assertEqual(actual.declared_frame_count,expected.declared_frame_count)
                self.assertEqual(actual.frames.keys(),expected.frames.keys())
                for index in expected.frames:
                    np.testing.assert_array_equal(actual.frames[index],expected.frames[index])
            self.assertEqual(cache.restarts,1)
            self.assertGreaterEqual(cache.cache_hits,3)
            self.assertLess(cache.decoded_frames,9+13+21+21+45)
            self.assertEqual(cache.diagnostics()['random_seeks'],0)

    def test_width_eviction_source_mutation_and_returned_array_isolation(self):
        with tempfile.TemporaryDirectory() as root:
            path=self.fixture(root)
            other=self.fixture(root,'other.avi')
            with SequentialReferenceFrameCache(maximum_bytes=30_000,maximum_open_sources=1) as cache:
                first=cache.decode(path,(3,)).frames[3]
                expected=first.copy()
                first[:]=0
                np.testing.assert_array_equal(cache.decode(path,(3,)).frames[3],expected)
                cache.decode(other,(4,))
                self.assertEqual(cache.diagnostics()['open_sources'],1)
                config=ReferenceMatchConfig(working_width=160)
                actual=cache.decode(path,(2,8),config=config)
                reference=decode_target_frames_sequential(path,(2,8),config=config)
                for index in actual.frames:
                    np.testing.assert_array_equal(actual.frames[index],reference.frames[index])
                self.assertLessEqual(cache.accounted_bytes,30_000)
                path.touch()
                with self.assertRaisesRegex(RuntimeError,'source changed'):
                    cache.decode(path,(8,))
            with self.assertRaisesRegex(RuntimeError,'closed'):
                cache.decode(other,(0,))


if __name__ == '__main__':
    unittest.main()
